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
# paper: RecDCL: Dual Contrastive Learning for Recommendation. WWW'24
# https://github.com/THUDM/RecDCL

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

class RecDCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(RecDCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['RecDCL'])
        self.n_layers = int(args['-n_layer'])
        self.gamma = float(args['-gamma']) # bt_coeff
        self.alpha = float(args['-alpha']) # poly_coeff
        self.beta = float(args['-beta']) # mon_coeff
        self.tau = float(args['-tau']) # momentum
        self.model = RecDCL_Encoder(self.data, self.emb_size, self.n_layers, self.gamma, self.alpha, self.beta, self.tau)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        while not early_stopping:
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, _ = batch
                rec_user_emb, rec_item_emb = model()
                
                user_emb, pos_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx]
                
                cl_loss = model.cal_cl_loss(user_idx, pos_idx)
                batch_loss = cl_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) 
                
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            measure, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class RecDCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, gamma, alpha, beta, tau):
        super(RecDCL_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        
        # constants
        self.all_bt_coeff = 1.
        self.a = 1.
        self.c = 1e-7 # poly_c
        self.e = 4. # degree
        
        self.bn = nn.BatchNorm1d(self.latent_size, affine=False).cuda()
        layers = []
        embs = str(self.latent_size) + '-' + str(self.latent_size) + '-' + str(self.latent_size)
        sizes = [self.latent_size] + list(map(int, embs.split('-')))
        for i in range(len(sizes)-2):
            layers.append(nn.Linear(sizes[i], sizes[i+1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        
        self.projector = nn.Sequential(*layers).cuda()
        self.predictor = nn.Linear(self.latent_size, self.latent_size).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
            'u_target_his': torch.randn((self.data.user_num, self.latent_size), requires_grad=False),
            'i_target_his': torch.randn((self.data.item_num, self.latent_size), requires_grad=False),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        self.all_embeddings = torch.mean(all_embeddings, dim=1)
        self.user_all_embeddings = self.all_embeddings[:self.data.user_num]
        self.item_all_embeddings = self.all_embeddings[self.data.user_num:]
        
        return self.user_all_embeddings, self.item_all_embeddings
    
    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()
    
    def bt(self, x, y):
        user_e = self.predictor(x)
        item_e = self.predictor(y)
        c = self.bn(user_e).T @ self.bn(item_e)
        c.div_(user_e.size()[0])
        
        # sum the cross-correlation matrix
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().div(self.latent_size)
        off_diag = self.off_diagonal(c).pow_(2).sum().div(self.latent_size)
        bt = on_diag + self.gamma * off_diag
        return bt
    
    def poly_feature(self, x):
        user_e = self.projector(x)
        xx = self.bn(user_e).T @ self.bn(user_e)
        poly = (self.a * xx + self.c) ** self.e
        return poly.mean().log()
    
    def loss_fn(self, p, z):  # cosine similarity
        return - torch.nn.functional.cosine_similarity(p, z.detach(), dim=-1).mean()
    
    def cal_cl_loss(self, user_idx, pos_idx):
        user_e, item_e = self.user_all_embeddings[user_idx], self.item_all_embeddings[pos_idx]
        
        with torch.no_grad():
            u_target, i_target = self.embedding_dict['u_target_his'].clone()[user_idx, :].cuda(), self.embedding_dict['i_target_his'].clone()[pos_idx, :].cuda()
            u_target.detach()
            i_target.detach()
            
            u_target = u_target * self.tau + user_e.data * (1. - self.tau)
            i_target = i_target * self.tau + item_e.data * (1. - self.tau)
            
            # update history embedding
            self.embedding_dict['u_target_his'][user_idx, :] = user_e.clone()
            self.embedding_dict['i_target_his'][pos_idx, :] = item_e.clone()
        
        user_e_n, item_e_n = torch.nn.functional.normalize(user_e, dim=-1), torch.nn.functional.normalize(item_e, dim=-1)
        user_e, item_e = self.predictor(user_e), self.predictor(item_e)
        
        # UIBT
        if self.all_bt_coeff == 0:
            bt_loss = 0.
        else:
            bt_loss = self.bt(user_e_n, item_e_n)
            
        # UUII
        if self.alpha == 0:
            poly_loss = 0.
        else:
            poly_loss = self.poly_feature(user_e_n) / 2 + self.poly_feature(item_e_n) / 2
        
        # BCL
        if self.beta == 0:
            mom_loss = 0.
        else:
            mom_loss = self.loss_fn(user_e, i_target) / 2 + self.loss_fn(item_e, u_target) / 2
            
        return self.all_bt_coeff * bt_loss + self.alpha * poly_loss + self.beta * mom_loss
