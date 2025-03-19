import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
import os
import numpy as np
import random
import pickle
import scipy.sparse as sp
from scipy.sparse.linalg import svds

# paper: Graph Signal Diffusion Model for Collaborative Filtering. SIGIR'24
# https://github.com/VinciZhu/GiffCF

seed = 0
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

class GiffCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(GiffCF, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['GiffCF'])
        self.ideal_weight = float(args['-ideal_weight'])
        self.noise_decay = float(args['-noise_decay'])
        self.model = GiffCF_Encoder(self.data, self.emb_size, self.ideal_weight, self.noise_decay)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        user_idx_list = list(range(self.data.user_num))
        while not early_stopping:
            #user_idx = list(range(self.data.user_num)) # full-training outperforms batch-training
            # batch_training for computational efficiency
            random.shuffle(user_idx_list)
            for i in range(self.data.user_num//self.batch_size):
                user_idx = user_idx_list[self.batch_size*i: self.batch_size*(i+1)]
                x_pred = model(user_idx)
                loss = torch.sum(torch.square(model.x_dense[user_idx] - x_pred), dim=-1) # batch
                reduced_loss = torch.mean(loss) # 1
                
                # Backward and optimize
                optimizer.zero_grad()
                reduced_loss.backward()
                optimizer.step()
            if self.batch_size*(i+1) < self.data.user_num:
                user_idx = user_idx_list[self.batch_size*(i+1):]
                x_pred = model(user_idx)
                loss = torch.sum(torch.square(model.x_dense[user_idx] - x_pred), dim=-1) # batch
                reduced_loss = torch.mean(loss) # 1
            
                # Backward and optimize
                optimizer.zero_grad()
                reduced_loss.backward()
                optimizer.step()
            if epoch % 100==0 and epoch>0:
                print('training:', epoch + 1, 'reduced_loss:', reduced_loss.item())
            with torch.no_grad():
                self.x_pred = []
                for i in range(self.data.user_num//self.batch_size):
                    self.x_pred_partial = model(list(range(self.batch_size*i,self.batch_size*(i+1))))
                    self.x_pred.append(self.x_pred_partial)
                if self.batch_size*(i+1)<self.data.user_num:
                    self.x_pred_partial = model(list(range(self.batch_size*(i+1),self.data.user_num)))
                    self.x_pred.append(self.x_pred_partial)
                self.x_pred = torch.cat(self.x_pred, dim=0)
            measure, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.x_pred = self.best_x_pred
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")

    def save(self):
        with torch.no_grad():
            self.best_x_pred = []
            for i in range(self.data.user_num//self.batch_size):
                self.x_pred_partial = self.model(list(range(self.batch_size*i,self.batch_size*(i+1))))
                self.best_x_pred.append(self.x_pred_partial)
            if self.batch_size*(i+1)<self.data.user_num:
                self.x_pred_partial = self.model(list(range(self.batch_size*(i+1),self.data.user_num)))
                self.best_x_pred.append(self.x_pred_partial)
            self.best_x_pred = torch.cat(self.best_x_pred, dim=0)
            #self.best_x_pred = self.model.forward(list(range(self.data.user_num)))

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = self.x_pred[u]
        return score.cpu().numpy()

class GiffCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, ideal_weight, noise_decay):
        super(GiffCF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.ideal_weight = ideal_weight
        self.noise_decay = noise_decay
        self.embedding_dict = self._init_model()
        self.noise_scale = 0.0 # default constant
        self.T = 3 # default constant
        self.alpha = 1.5 # default constant
        self.ideal_cutoff = 200 # default constant
        self.dropout = 0.5 # default constant
        self.t = torch.linspace(0, self.T, self.T+1).float().cuda()
        
        # item-item similarity matrix
        self.adj_mat = data.interaction_mat
        adj_mat = data.interaction_mat
        user_deg = np.array(adj_mat.sum(axis=1)) # m
        item_deg = np.array(adj_mat.sum(axis=0)) # n
        user_inv = np.power(user_deg, -0.25).flatten()
        user_inv[np.isinf(user_inv)] = 0.
        user_mat = sp.diags(user_inv)
        item_inv = np.power(item_deg, -0.5).flatten()
        item_inv[np.isinf(item_inv)] = 0.
        item_mat = sp.diags(item_inv)
        adj_right = (user_mat.dot(adj_mat)).dot(item_mat) # m*n
        self.adj_right = TorchGraphInterface.convert_sparse_mat_to_tensor(adj_right).cuda() # m*n
        # self.adj_left = self.adj_right.transpose(0,1)
        # self.adj_left = TorchGraphInterface.convert_sparse_mat_to_tensor(adj_right.T).cuda() # n*m
        x_sparse = TorchGraphInterface.convert_sparse_mat_to_tensor(adj_mat).cuda() # m*n
        self.x_dense = x_sparse.to_dense()
        
        # ideal low-pass
        self.eigen = self.compute_eigen(adj_right, self.ideal_cutoff)
        self.eigen_val = torch.tensor(self.eigen['values'], dtype=torch.float32).cuda() # cutoff
        self.eigen_vec = torch.tensor(self.eigen['vectors'], dtype=torch.float32).cuda() # cutoff*n
    
    def compute_eigen(self, adj_right, cutoff):
        _, values, vectors = svds(adj_right, k=cutoff)
        idx = np.argsort(values)[::-1]
        values = values[idx] ** 2 # cutoff
        vectors = vectors[idx] # cutoff*n
        return {'cutoff': cutoff, 'values': values, 'vectors': vectors}
        
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
            'time_in': nn.Parameter(initializer(torch.empty(20, 20))), # default constant hidden size
            'time_out': nn.Parameter(initializer(torch.empty(20, 1))), # default constant output size
            'embed_mixer_in': nn.Parameter(initializer(torch.empty(3, 2))), # default constant hidden size
            'embed_mixer_out': nn.Parameter(initializer(torch.empty(2, 1))), # default constant output size
            'score_mixer_in': nn.Parameter(initializer(torch.empty(4, 2))), # default constant hidden size
            'score_mixer_out': nn.Parameter(initializer(torch.empty(2, 1))), # default constant output size
        })
        return embedding_dict
    
    def prop(self, x):
        x_prop = torch.sparse.mm(self.adj_right, x.adjoint()) # m*batch
        x_prop = torch.sparse.mm(self.adj_right.transpose(0,1), x_prop) # n*batch
        x_prop = x_prop.transpose(0,1) # batch*n
        return x_prop / self.eigen_val[0]
    
    def ideal(self, x, cutoff=None):
        eigen_vec = self.eigen_vec[:cutoff] if cutoff is not None else self.eigen_vec # cutoff*n
        x_ideal = torch.matmul(x, eigen_vec.transpose(0,1)) # batch*cutoff
        x_ideal = torch.matmul(x_ideal, eigen_vec) # batch*n
        return x_ideal
    
    def smooth(self, x):
        x_smooth = self.ideal_weight * self.ideal(x) + self.prop(x) # batch*n
        return x_smooth / (1+self.ideal_weight)
    
    def filter_(self, x, Ax, t):
        return self.alpha * t / self.T * (Ax - x) + x # F_t x, batch*n
    
    def sigma(self, t):
        return self.noise_scale * self.noise_decay ** (self.T - t)
    
    def denoise(self, z_t, c, Ac, t, training=False):
        t = torch.broadcast_to(t, (z_t.shape[0],1)) # batch*1
        x_pred = self.denoiser(z_t, c, Ac, t, training)
        return x_pred
    
    def Timestep(self, embed_dim, n_steps, t, max_wavelength=10000.0):
        timescales = np.power(max_wavelength, -np.arange(0, embed_dim, 2)/embed_dim) # embed_dim/2
        timesteps = np.arange(n_steps+1) # n_step+1
        angles = timesteps[:, np.newaxis] * timescales[np.newaxis, :] # (n_step+1)*(embed_dim/2)
        sinusoids = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1) # (n_step+1)*embed_dim
        sinusoids = torch.tensor(sinusoids, dtype=torch.float32).cuda() # (n_step+1)*embed_dim
        return sinusoids[torch.reshape(t, [-1]).long()] # batch*embed_dim
    
    def TimeEmbed(self, hidden_dim, out_dim, n_steps, t):
        e = self.Timestep(hidden_dim, n_steps, t)
        hidden = torch.matmul(e, self.embedding_dict['time_in']) # batch*20
        hidden = torch.nn.SiLU()(hidden)
        out = torch.matmul(hidden, self.embedding_dict['time_out']) # batch*1
        return out
    
    def embedMixer(self, inputs):
        x = torch.stack(inputs, dim=-1) # batch*latent_size*3
        hidden = torch.matmul(x, self.embedding_dict['embed_mixer_in']) # batch*latent_size*2
        hidden = torch.nn.SiLU()(hidden)
        out = torch.matmul(hidden, self.embedding_dict['embed_mixer_out']) # batch*latent_size*1
        return out.squeeze(dim=-1) # batch*latent_size

    def scoreMixer(self, inputs):
        x = torch.stack(inputs, dim=-1) # batch*latent_size*4
        hidden = torch.matmul(x, self.embedding_dict['score_mixer_in']) # batch*latent_size*2
        hidden = torch.nn.SiLU()(hidden)
        out = torch.matmul(hidden, self.embedding_dict['score_mixer_out']) # batch*latent_size*1
        return out.squeeze(dim=-1) # batch*latent_size
    
    def denoiser(self, z_t, c, Ac, t, training=None):
        norm_ord = 1 # default constant
        t_embed1 = self.TimeEmbed(20, 1, self.T, t).repeat(1, self.latent_size) # batch*latent_size
        t_embed2 = self.TimeEmbed(20, 1, self.T, t).repeat(1, self.data.item_num) # batch*n
        z_embed = torch.matmul(z_t / self.data.item_num, self.embedding_dict['item_emb']) # batch*latent_size
        c_embed = torch.matmul(c / norm_ord, self.embedding_dict['item_emb']) # batch*latent_size
        
        x_embed = self.embedMixer([z_embed, c_embed, t_embed1]) # batch*latens_size
        x_mid = torch.matmul(x_embed, self.embedding_dict['item_emb'].transpose(0,1)) # batch*n
        x_pred = self.scoreMixer([x_mid, c, Ac, t_embed2]) # batch*n
        return x_pred
    
    def forward(self, user_idx):
        x = self.x_dense[user_idx] # batch*n
        if self.training:
            t = torch.randint(1, self.T+1, ((x.shape)[0], 1), dtype=torch.int32).cuda() # expectation of steps, batch*1
            Ax = self.smooth(x) # batch*n
            z_t = self.filter_(x, Ax, t) # F_t\times x, batch*n
            if self.noise_scale > 0.0:
                eps = torch.normal(size=x.shape)
                z_t += self.sigma(t) * eps # Eq.(8)
            c = torch.nn.Dropout(self.dropout)(x) # batch*n
            Ac = self.smooth(c) # batch*n
            x_pred = self.denoise(z_t, c, Ac, t, self.training) # batch*n
            return x_pred
        else:
            Ax = self.smooth(x) # batch*n
            z_t = self.filter_(x, Ax, self.t[-1]) # F_t\times x, batch*n
            for i in range(len(self.t)-1, 0, -1):
                t, s = self.t[i], self.t[i-1] # 1
                x_pred = self.denoise(z_t, x, Ax, t) # batch*n
                Ax_pred = self.smooth(x_pred) # batch*n
                z_s_pred = self.filter_(x_pred, Ax_pred, s) # batch*n
                if self.noise_decay > 0.0:
                    z_t_pred = self.filter_(x_pred, Ax_pred, t) # batch*n
                    z_t = z_s_pred + self.noise_decay ** (t-s) * (z_t-z_t_pred) # batch*n
                else:
                    z_t = z_s_pred # batch*n
            x_pred = z_t # batch*n
            return x_pred
            
    
    
    
        


