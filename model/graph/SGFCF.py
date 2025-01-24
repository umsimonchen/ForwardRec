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
import gc
# paper: SGFCF: How Powerful is Graph Filtering for Recommendation. KDD'24
# https://github.com/tanatosuu/sgfcf

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

class SGFCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SGFCF, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SGFCF'])
        self.frequency = int(args['-frequency'])
        self.alpha = float(args['-alpha'])
        self.beta = float(args['-beta'])
        self.beta_1 = float(args['-beta_1'])
        self.beta_2 = float(args['-beta_2'])
        self.eps = float(args['-eps']) # reciprocal of the exponent
        self.gamma = float(args['-gamma'])
        self.model = SGFCF_Encoder(self.data, self.emb_size, self.frequency, self.alpha, self.beta_1, self.beta_2, self.beta, self.gamma, self.eps)
        
    def train(self):
        # Training-free model
        model = self.model.cuda()
        self.score = model()
        _, _ = self.fast_evaluation(0)
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")
            
    def save(self):
        # Training-free model
        pass

    def predict(self, u):
        u = self.data.get_user_id(u)
        return self.score[u].cpu().numpy()

class SGFCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, frequency, alpha, beta_1, beta_2, beta, eps, gamma):
        super(SGFCF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.frequency = frequency
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta = beta
        self.eps = eps
        self.gamma = gamma
        self.embedding_dict = self._init_model()
        self.freq_matrix = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda().to_dense()

    def _init_model(self):
        # Training-free model
        pass
    
    # map homo_ratio into [beta1, beta2]
    def individual_weight(self, value, homo_ratio):
        y_min, y_max = self.beta_1, self.beta_2
        x_min, x_max = homo_ratio.min(), homo_ratio.max()
        # equivalent to: y_min + (x-x_min)*(y_max-y_min)/(x_max-x_min)
        homo_weight = (y_max - y_min) / (x_max - x_min) * homo_ratio + (x_max*y_min - x_min*y_max) / (x_max - x_min)
        homo_weight = homo_weight.unsqueeze(1)
        return value.pow(homo_weight)
    
    # monomial filter beta
    def weight_func(self, value):
        return value**self.beta
        
    def forward(self):
        homo_ratio_user, homo_ratio_item = [], []
        users = list(self.data.training_set_u.keys())
        items = list(self.data.training_set_i.keys())
        
        for u in users:
            if len(self.data.training_set_u[u]) > 1:
                i = []
                for item in list(self.data.training_set_u[u].keys()):
                    i.append(self.data.item[item])

                inter_items = self.freq_matrix[:, i].t() # n~ * m: select the column of bought items and transpose as item representation
                inter_items[:, self.data.user[u]] = 0 # n~ * m: w/o u as bridge
                connect_matrix = inter_items.mm(inter_items.t()) # n~ * n~
      		
                #connect_matrix=connect_matrix+connect_matrix.mm(connect_matrix)+connect_matrix.mm(connect_matrix).mm(connect_matrix)
      
                size = inter_items.shape[0] 
                #homo_ratio, reachable within 2 steps w/o u, minus self-connection and divide due to symmetry
                ratio_u = ((connect_matrix != 0).sum().item() - (connect_matrix.diag() != 0).sum().item()) / (size * (size - 1)) 
                homo_ratio_user.append(ratio_u)
            else:
                homo_ratio_user.append(0)
        
        for i in items:
            if len(self.data.training_set_i[i]) > 1:
                u = []
                for user in list(self.data.training_set_i[i].keys()):
                    u.append(self.data.user[user])
                    
                inter_users = self.freq_matrix[u] # m~ * n: select the column of buying users and transpose as the user representation
                inter_users[:, self.data.item[i]] = 0 # m~ * n: w/o i as bridge
                connect_matrix = inter_users.mm(inter_users.t())  # m~ * m~
        
                #connect_matrix=connect_matrix+connect_matrix.mm(connect_matrix)+connect_matrix.mm(connect_matrix).mm(connect_matrix)
        
                size = inter_users.shape[0]
                #homo_ratio, reachable within 2 steps w/o u, minus self-connection and divide due to symmetry
                ratio_i = ((connect_matrix != 0).sum().item() - (connect_matrix.diag() != 0).sum().item()) / (size * (size - 1))
                homo_ratio_item.append(ratio_i)
            else:
                homo_ratio_item.append(0)
        
        homo_ratio_user = torch.Tensor(homo_ratio_user).cuda()
        homo_ratio_item = torch.Tensor(homo_ratio_item).cuda()
        
        # Generalized Graph Normalization (G^2N)
        D_u = 1 / (self.freq_matrix.sum(1) + self.alpha).pow(self.eps)
        D_i = 1 / (self.freq_matrix.sum(0) + self.alpha).pow(self.eps)
        D_u[D_u==float('inf')] = 0
        D_i[D_i==float('inf')] = 0
        self.norm_freq_matrix = D_u.unsqueeze(1) * self.freq_matrix * D_i
        
        # Singular decomposition
        U, value, V = torch.svd_lowrank(self.norm_freq_matrix, q=self.frequency, niter=30)
        # Sharpen spectrum
        value = value / value.max()
        
        # # Individualized Graph Filtering (IGF) by elemnet-wise multiplication
        # rate_matrix = (U[:, :self.frequency] * self.individual_weight(value[:self.frequency], homo_ratio_user)).mm \
        #     ((V[:, :self.frequency] * self.individual_weight(value[:self.frequency], homo_ratio_item)).t())
        
        # w/o IGF
        rate_matrix = (U[:, :self.frequency] * self.weight_func(value[:self.frequency])).mm \
            ((V[:, :self.frequency] * self.weight_func(value[:self.frequency])).t())
        
        # remove used variable to save memory
        del homo_ratio_user, homo_ratio_item
        gc.collect()
        torch.cuda.empty_cache()
        
        rate_matrix = rate_matrix / (rate_matrix.sum(1).unsqueeze(1)) # user-based norm
        self.norm_freq_matrix = self.norm_freq_matrix.mm(self.norm_freq_matrix.t()).mm(self.norm_freq_matrix) # contain all frequencies
        self.norm_freq_matrix = self.norm_freq_matrix / (self.norm_freq_matrix.sum(1).unsqueeze(1)) # user-based norm
        rate_matrix = (rate_matrix + self.gamma * self.norm_freq_matrix).sigmoid()
        rate_matrix = rate_matrix - self.freq_matrix*1000 # remove the bought items
        
        # remove used variable to save memory
        del U, V, value, D_u, D_i, self.freq_matrix, self.norm_freq_matrix
        gc.collect()
        torch.cuda.empty_cache()
        
        return rate_matrix


