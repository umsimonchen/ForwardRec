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
import math
# paper: MultiVAE: Variational autoencoders for collaborative filtering. WWW'18
# https://github.com/dawenl/vae_cf

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

class MultiVAE(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(MultiVAE, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['MultiVAE'])
        self.layers = eval(args['-mlp_hidden_size'])
        self.drop_out = float(args['-dropout_prob'])
        self.anneal_cap = float(args['-anneal_cap'])
        self.total_anneal_steps = int(args['-total_anneal_steps'])
        self.model = MultiVAE_Encoder(self.data, self.emb_size, self.layers, self.drop_out, self.anneal_cap, self.total_anneal_steps)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        total_batch = math.ceil(self.data.user_num/self.batch_size)
        while not early_stopping:
            # full training
            # user_idx = list(range(self.data.user_num))
            # z, mu, logvar = model(user_idx)
            
            # if self.total_anneal_steps > 0:
            #     anneal = min(self.anneal_cap, 1.0 * (epoch+1) / self.total_anneal_steps)
            # else:
            #     anneal = self.anneal_cap
            
            # # KL loss
            # kl_loss = (
            #     -0.5
            #     * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            #     * anneal
            # )
    
            # # CE loss
            # ground_true = model.torch_norm_inter[user_idx]
            # ce_loss = -(torch.nn.functional.log_softmax(z, 1) * ground_true).sum(1).mean()
            # total_loss = ce_loss + kl_loss
            
            # batch training
            total_loss = 0
            for i in range(total_batch):
                user_idx = list(range(i*self.batch_size,min((i+1)*self.batch_size, self.data.user_num)))
                z, mu, logvar = model(user_idx)
                
                if self.total_anneal_steps > 0:
                    anneal = min(self.anneal_cap, 1.0 * (epoch+1) / self.total_anneal_steps)
                else:
                    anneal = self.anneal_cap
                
                # KL loss
                kl_loss = (
                    -0.5
                    * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                    * anneal
                )
        
                # CE loss
                ground_true = model.torch_norm_inter[user_idx]
                ce_loss = -(torch.nn.functional.log_softmax(z, 1) * ground_true).sum(1).mean()
                total_loss += ce_loss + kl_loss

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print('training:', epoch + 1, 'batch:', str(i+1)+'/'+str(total_batch), 'loss:', total_loss.item())
            with torch.no_grad():
                # full evaluation
                # user_idx = list(range(self.data.user_num))
                # self.all_scores, _, _ = model(user_idx).cpu().numpy()
                
                # batch evalution
                self.all_scores = []
                for i in range(total_batch):
                    user_idx = list(range(i*self.batch_size,min((i+1)*self.batch_size, self.data.user_num)))
                    scores, _, _ = model(user_idx)
                    self.all_scores.append(scores.cpu().numpy())
                self.all_scores = np.concatenate(self.all_scores, axis=0)
            measure, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.all_scores = self.best_scores
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")
        # record training loss        
        # with open('training_record','wb') as fp:
        #     pickle.dump([record_list, loss_list], fp)
        
    def save(self):
        with torch.no_grad():
            self.best_scores = self.all_scores

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = self.all_scores[u]
        return score

class MultiVAE_Encoder(nn.Module):
    def __init__(self, data, emb_size, layers, drop_out, anneal_cap, total_anneal_steps):
        super(MultiVAE_Encoder, self).__init__()
        self.data = data
        self.lat_dim = emb_size
        self.layers = layers
        self.drop_out = drop_out
        self.anneal_cap = anneal_cap
        self.total_anneal_steps = total_anneal_steps
        self.norm_inter = data.norm_inter
        self.torch_norm_inter = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_inter).cuda().to_dense()
        self.encode_layer_dims = [self.data.item_num] + self.layers + [self.lat_dim]
        self.decode_layer_dims = [int(self.lat_dim / 2)] + self.encode_layer_dims[::-1][1:]

        self.encoder = self.mlp_layers(self.encode_layer_dims)
        self.decoder = self.mlp_layers(self.decode_layer_dims)
    
    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=0.01)
            return mu + epsilon * std
        else:
            return mu

    def forward(self, user_idx=None):
        if user_idx == None:
            user_idx = list(range(self.data.user_num))
        h = torch.nn.functional.normalize(self.torch_norm_inter[user_idx])
        h = torch.nn.functional.dropout(h, self.drop_out, training=self.training)
        h = self.encoder(h)

        mu = h[:, : int(self.lat_dim / 2)]
        logvar = h[:, int(self.lat_dim / 2) :]

        z = self.reparameterize(mu, logvar)
        z = self.decoder(z)
        return z, mu, logvar


