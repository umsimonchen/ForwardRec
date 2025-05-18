import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
import os
import numpy as np
import random
import pickle
import scipy.sparse as sp
from sparsesvd import sparsesvd
from torchdiffeq import odeint
import math
import sys
# paper: Blurring-Sharpening Process Models for Collaborative Filtering. SIGIR'23
# https://github.com/jeongwhanchoi/BSPM/tree/main

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

class BSPM(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(BSPM, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['BSPM'])
        self.idl_beta = float(args['-idl_beta'])
        self.sharpen_T = int(args['-sharpen_T']) # end
        self.sharpen_K = int(args['-sharpen_K']) # steps
        self.sharpen_solver = str(args['-sharpen_solver'])
        self.model = BSPM_Encoder(self.data, self.emb_size, self.idl_beta, self.sharpen_T, self.sharpen_K,  self.sharpen_solver)

    def train(self):
        model = self.model.cuda()
        
        # whole dataset training
        # self.score = model() 
        
        # training by batch-slice in case of OOM on large datasets
        total_batch = math.ceil(self.data.user_num/self.batch_size)
        lst_score = []
        for i in range(total_batch):
            batch_test = model.convert_sp_mat_to_sp_tensor(model.adj_mat[i*self.batch_size:(i+1)*self.batch_size]).cuda()
            lst_score.append(model(batch_test).cpu().numpy())
            print("Finished Batch:%d / %d."%(i+1,total_batch))
        self.score = np.concatenate(lst_score, axis=0)
        
        measure, early_stopping = self.fast_evaluation(0)
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")
        sys.exit() # tested by fast evaluation

    def save(self):
        with torch.no_grad():
            self.best_score = self.score

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = self.score[u]
        return score

class BSPM_Encoder(nn.Module):
    def __init__(self, data, emb_size, idl_beta, sharpen_T, sharpen_K, sharpen_solver):
        super(BSPM_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.idl_beta = idl_beta
        self.final_sharpening = True # default constant, True: Early Merge; False: Late Merge
        self.t_point_combination = False # default constant, True: Use the combination of t points

        # row-normalization
        self.adj_mat = data.interaction_mat # m*n
        rowsum = np.array(self.adj_mat.sum(axis=1)) # m
        d_inv = np.power(rowsum, -0.5).flatten() 
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv) # m*m
        norm_adj = d_mat.dot(self.adj_mat) # m*n
        
        # column-normalization
        colsum = np.array(self.adj_mat.sum(axis=0)) # n
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv) # n*n
        self.d_mat_i = d_mat # n*n
        self.d_mat_i_inv = sp.diags(1/d_inv) # n*n
        norm_adj = norm_adj.dot(d_mat) # m*n 
        self.norm_adj = norm_adj.tocsc() # m*n 
        del norm_adj, d_mat
        ut, s, self.vt = sparsesvd(self.norm_adj, self.latent_size) # m*k, k*k, k*n
        del ut
        del s
        
        # linear filter
        linear_Filter = self.norm_adj.T @ self.norm_adj # \tilde{P}, n*n
        self.linear_Filter = self.convert_sp_mat_to_sp_tensor(linear_Filter).to_dense().cuda() # n*n
        left_mat = self.d_mat_i @ self.vt.T # V^{-1/2}U, n*k
        right_mat = self.vt @ self.d_mat_i_inv # U^{\top}V^{1/2}, k*n
        self.left_mat, self.right_mat = torch.FloatTensor(left_mat).cuda(), torch.FloatTensor(right_mat).cuda() # n*k, k*n
        del left_mat
        del right_mat
        
        
        # time-step
        idl_T = 1 # default constant
        idl_K = 1 # default constant
        blur_T = 1 # default constant
        blur_K = 1 # default constant
        sharpen_T = sharpen_T
        sharpen_K = sharpen_K
        self.idl_times = torch.linspace(0, idl_T, idl_K+1).float().cuda()
        self.blurring_times = torch.linspace(0, blur_T, blur_K+1).float().cuda()
        self.sharpening_times = torch.linspace(0, sharpen_T, sharpen_K+1).float().cuda()
        
        # solver
        self.idl_solver = 'euler' # default constant
        self.blur_solver = 'euler' # default constant
        self.sharpen_solver = sharpen_solver
    
    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))
    
    def sharpenFunction(self, t, r):
        out = r @ self.linear_Filter
        return -out
    
    def forward(self, batch_test=None):
        if batch_test == None:
            self.convert_sp_mat_to_sp_tensor(self.adj_mat).cuda() # m*nself.sparse_adj_norm
        
        with torch.no_grad():
            idl_out = torch.sparse.mm(batch_test, self.left_mat @ self.right_mat)
            blurred_out = torch.sparse.mm(batch_test, self.linear_Filter)
            del batch_test
            
            if self.final_sharpening == True:
                sharpened_out = odeint(func=self.sharpenFunction, y0=self.idl_beta*idl_out + blurred_out, t=self.sharpening_times, method=self.sharpen_solver)
            else: 
                sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out, t=self.sharpening_times, method=self.sharpen_solver)
                
        if self.t_point_combination == True:
            U_2 =  torch.mean(torch.cat([blurred_out.unsqueeze(0),sharpened_out[1:,...]],axis=0),axis=0)
        else:
            U_2 = sharpened_out[-1]
            del sharpened_out

        if self.final_sharpening == True:
            ret = U_2
        else:
            ret = self.idl_beta * idl_out + U_2
        return ret
        
    
    
    
        


