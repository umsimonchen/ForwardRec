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
import scipy as sp
import math

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

class DiffGraph(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(DiffGraph, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['DiffGraph'])
        self.steps = int(args['-steps'])
        self.noise_scale = float(args['-noise_scale'])
        self.noise_min = float(args['-noise_min'])
        self.noise_max = float(args['-noise_max'])
        self.model = DiffGraph_Encoder(self.data, self.emb_size, self.steps, self.noise_scale, self.noise_min, self.noise_max)
        
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        while not early_stopping:
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
            
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                print('training:', epoch + 1, 'loss:', batch_loss.item())
            with torch.no_grad():
                self.rec_ui = model.evaluation()
            _, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.rec_ui = self.best_rec_ui
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class DiffGraph_Encoder(nn.Module):
    def __init__(self, data, emb_size, steps, noise_scale, noise_min, noise_max):
        super(DiffGraph_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.steps = steps
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict
    

    def forward(self):
        user_variance, user_mean = torch.var_mean(self.embedding_dict['user_emb'], dim=0)
        item_variance, item_mean = torch.var_mean(self.embedding_dict['item_emb'], dim=0)
        
        for 
        
        
        
    
    
    
        


