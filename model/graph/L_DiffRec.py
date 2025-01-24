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
import math
import pickle
import scipy as sp
from kmeans_pytorch import kmeans
# paper: Diffusion Recommender Model - Latent Variant. SIGIR'23
# https://github.com/YiyanXu/DiffRec/tree/main/L-DiffRec
# Require trained item embeddings from LightGCN.

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

class L_DiffRec(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(L_DiffRec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['L_DiffRec'])
        self.steps = int(args['-steps'])
        self.noise_scale = float(args['-noise_scale'])
        self.noise_min = float(args['-noise_min'])
        self.noise_max = float(args['-noise_max'])
        self.n_cate = int(args['-n_cate'])
        self.model = L_DiffRec_Encoder(self.data, self.emb_size, self.steps, self.noise_scale, self.noise_min, self.noise_max, self.n_cate)
        
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        while not early_stopping:
            batch_idx = torch.randperm(self.data.user_num)[:self.batch_size]
                
            losses = model(batch_idx)
            
            elbo = losses["loss"].mean()
            anneal = 0.0003 # customized as DiffRec
            lamda = 10 # customized as DiffRec
            vae_loss = losses["vae_loss"] + anneal * losses["vae_kl"]
            loss = elbo + lamda * vae_loss # reweight required
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('training:', epoch + 1, 'loss:', loss.item())
            with torch.no_grad():
                self.rec_ui = model.evaluation()
            _, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.rec_ui = self.best_rec_ui
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")

    def save(self):
        with torch.no_grad():
            self.best_rec_ui = self.rec_ui

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = self.rec_ui[u]
        return score.cpu().numpy()

class L_DiffRec_Encoder(nn.Module):
    def __init__(self, data, emb_size, steps, noise_scale, noise_min, noise_max, n_cate):
        super(L_DiffRec_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.steps = steps
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.n_cate = n_cate
        self.norm_adj = data.norm_adj
        self.binary = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda().to(torch.float64).to_dense()
        
        # eigendecomposition is not the optimal solution (suggesting pre-trained embedding), but used here for brevity
        eigenvalues, eigenvectors = sp.sparse.linalg.eigsh(self.norm_adj, k=self.latent_size, which='SA', tol=1e-2)
        if self.n_cate > 1:
            self.cluster_ids, _ = kmeans(X=self.item_emb, num_clusters=self.n_cate, distance='euclidean')
            category_idx = []
            for i in range(self.n_cate):
                idx = np.argwhere(self.cluster_ids.numpy()==i).squeeze().tolist()
                category_idx.append(torch.tensor(idx, dtype=int)) # find the index of items that belong to category i
            self.category_idx = category_idx
            self.category_map = torch.cat(tuple(category_idx), dim=-1) # [item_num]
            self.category_len = [len(self.category_idx[i]) for i in range(n_cate)]
            self.reverse_map = {self.category_map[i]:i for i in range(len(self.category_map))} # make mapping dictionary
        
        self.history_num_per_term = 10 # only recording for analysis
        self.Lt_history = torch.zeros(self.steps, self.history_num_per_term, dtype=torch.float64)
        self.Lt_count = torch.zeros(self.steps, dtype=torch.int)  
        if self.noise_scale != 0.:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()
            self.betas[0] = 0.00001 # since beta is fixed, small constant can prevent overfitting according to Noneequilibrium Thermodynamics
        self.calculate_for_diffusion()
        
        self.weight_dict = self._init_model()
    
    def get_betas(self): #linear-variance used
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        #betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1-alpha_bar[i]/alpha_bar[i-1], 0.999)) # 0.999 is set as max beta
        return np.array(betas)
    
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus__cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) * (1.0 - self.alphas_cumprod)
    
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        weight_dict = nn.ParameterDict({
            'w_in': nn.Parameter(initializer(torch.empty(self.latent_size*self.n_cate + self.latent_size, self.latent_size, dtype=torch.float64))),
            'b_in': nn.Parameter(initializer(torch.empty(1, self.latent_size, dtype=torch.float64))),
            'w_out': nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size*self.n_cate, dtype=torch.float64))),
            'b_out': nn.Parameter(initializer(torch.empty(1, self.latent_size*self.n_cate, dtype=torch.float64))),
        })
        
        if self.n_cate == 1:
            weight_dict['vae_encoder'] = nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size*2, dtype=torch.float64)))
            weight_dict['vae_decoder'] = nn.Parameter(initializer(torch.empty(self.latent_size, self.data.item_num, dtype=torch.float64)))
        else:
            for i in range(self.n_cate):
                weight_dict['vae_encoder%d'%i] = nn.Parameter(initializer(torch.empty(self.category_len[i], self.latent_size*2, dtype=torch.float64)))
            # suppose one-layer decoder
            weight_dict['vae_decoder'] = nn.Parameter(initializer(torch.empty(self.latent_size*self.n_cate, self.data.item_num, dtype=torch.float64)))
        return weight_dict
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float64) / half).cuda()
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
        return embedding
    
    def DNN_model(self, x_t, ts):
        time_emb = self.timestep_embedding(ts, self.latent_size)
        h = torch.cat([x_t, time_emb], dim=1)
        model_output = torch.matmul(h, self.weight_dict['w_in']) + self.weight_dict['b_in']
        model_output = torch.tanh(model_output)
        model_output = torch.matmul(model_output, self.weight_dict['w_out']) + self.weight_dict['b_out']
        model_output = torch.tanh(model_output)

        return model_output
    
    def VAE_model_encode(self, batch, training=True):
        if self.n_cate == 1:
            hidden = torch.matmul(batch, self.weight_dict['vae_encoder'])
            hidden = torch.tanh(hidden)
            mu = hidden[:, :self.latent_size]
            logvar = hidden[:, self.latent_size:]
        
        else:
            batch_cate = []
            for i in range(self.n_cate):
                batch_cate.append(batch[:, self.category_idx[i]]) # divide the bought items into category for every sample in batch
            latent_mu = []
            latent_logvar = []
            for i in range(self.n_cate):
                hidden = torch.matmul(batch_cate[i], self.weight_dict['vae_encoder%d'%i])
                hidden = torch.tanh(hidden)
                latent_mu.append(hidden[:, :self.latent_size])
                latent_logvar.append(hidden[:, :self.latent_size])
                
            mu = torch.cat(tuple(latent_mu), dim=1)
            logvar = torch.cat(tuple(latent_logvar), dim=1)
            batch = torch.cat(tuple(batch_cate), dim=1)
        
        if training:
            latent = mu + torch.mul(torch.exp(0.5*logvar), torch.randn_like(logvar))
        else:
            latent = mu
        kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        return batch, latent, kl_divergence
        
    def VAE_model_decode(self, batch):
        # suppose one-layer decoder
        output = torch.matmul(batch, self.weight_dict['vae_decoder'])
        output = torch.tanh(output)
        
        return output
                
    
    def sample_timesteps(self, batch_size, uniform_prod=0.001): #importance-sampling used
        if not (self.Lt_count == self.history_num_per_term).all():
            t = torch.randint(0, self.steps, (batch_size,)).long()
            pt = torch.ones_like(t).float()
        
        else:
            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prod # uniform probability 0.001
            pt_all += uniform_prod / len(pt_all)
            
            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)
        
        return t.cuda(), pt.cuda()
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)*x_start + \
                self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape))

    def forward(self, batch_idx):
        batch = self.binary[batch_idx]
        batch_cate, batch_latent, vae_kl = self.VAE_model_encode(batch)
        
        x_start = batch_latent
        batch_size = x_start.size(0)
        ts, pt = self.sample_timesteps(batch_size)
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        terms = {}
        model_output = self.DNN_model(x_t, ts)
        loss = self.mean_flat((batch_latent - model_output)**2)
        
        # reweight is True
        weight = self.SNR(ts-1) - self.SNR(ts)
        weight = torch.where((ts==0), 1.0, weight)
        terms["loss"] = weight * loss
        
        # update Lt_history 
        for t, loss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                self.Lt_count[t] += 1
        
        terms["loss"] /= pt
        
        batch_recon = self.VAE_model_decode(model_output)
        terms['vae_loss'] = -torch.mean(torch.sum(nn.functional.log_softmax(batch_recon, 1) * batch_cate, 1))
        terms['vae_kl'] = vae_kl
        
        return terms
    
    def evaluation(self): # default sampled step is 0 and noise_scale is not 0
        _, x_t, _ = self.VAE_model_encode(self.binary) # step is 0 as default
        indices = list(range(self.steps))[::-1]
        
        if self.noise_scale == 0:
            for i in indices:
                t = torch.tensor([i] * x_t.shape[0]).cuda()
                x_t = self.DNN_model(x_t, t)
            return x_t
        
        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).cuda()
            out = self.p_mean_variance(x_t, t)
            
            # only use mean for inference but no noise
            x_t = out["mean"]
        
        x_t = self.VAE_model_decode(x_t)
    
        return x_t
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + \
                          self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
        
    def p_mean_variance(self, x, t):
        model_output = self.DNN_model(x, t)
        
        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped
        
        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        pred_xstart = model_output # original data is used as default 
        model_mean, model_variance, model_log_variance = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        
        return {"mean": model_mean, "variance": model_variance, "log_variance": model_log_variance, "pred_xstart": pred_xstart}
    
    def SNR(self, t):
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    
    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
    
        


