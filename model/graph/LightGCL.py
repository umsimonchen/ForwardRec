import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor
import os
import numpy as np
import random
import pickle
# paper: LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation ICLR'23
# https://github.com/HKUDS/LightGCL

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

class LightGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['LightGCL'])
        n_layers = int(args['-n_layer'])
        drop_rate = float(args['-droprate'])
        temp = float(args['-temp'])
        self.lambda_1 = float(args['-lambda_1'])
        self.model = LightGCL_Encoder(self.data, self.emb_size, n_layers, drop_rate, temp)

    def train(self):
        record_list = []
        loss_list = []
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        while not early_stopping:
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, model.embedding_dict['user_emb'], model.embedding_dict['item_emb'])/self.batch_size
                cl_loss = model.cal_cl_loss(user_idx, pos_idx+neg_idx)
                total_loss = batch_loss + self.lambda_1 * cl_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
                
            measure, early_stopping = self.fast_evaluation(epoch)
            record_list.append(measure)
            loss_list.append(batch_loss.item())
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

class LightGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, drop_rate, temp):
        super(LightGCL_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_inter = data.norm_inter
        self.drop_rate = drop_rate
        self.temp = temp
        self.embedding_dict = self._init_model()
        self.sparse_norm_inter = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_inter).cuda()
        
        self.E_u_list = [None] * (self.layers+1)
        self.E_i_list = [None] * (self.layers+1)
        self.E_u_list[0] = self.embedding_dict['user_emb']
        self.E_i_list[0] = self.embedding_dict['item_emb']
        self.Z_u_list = [None] * (self.layers+1)
        self.Z_i_list = [None] * (self.layers+1)
        self.G_u_list = [None] * (self.layers+1)
        self.G_i_list = [None] * (self.layers+1)
        self.G_u_list[0] = self.embedding_dict['user_emb']
        self.G_i_list[0] = self.embedding_dict['item_emb']
        self.act = nn.LeakyReLU(0.5)
        
        self.E_u = None
        self.E_i = None
        
        svd_u, s, svd_v = torch.svd_lowrank(self.sparse_norm_inter, q=5) # default constant

        u_mul_s = svd_u @ torch.diag(s)
        v_mul_s = svd_v @ torch.diag(s)

        self.ut = svd_u.T
        self.vt = svd_v.T
        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        del s

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def sparse_dropout(self):
        dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.sparse_adjacency_matrix_R(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()
    
    def forward(self):
        for layer in range(1, self.layers+1):
            # GNN propagation
            self.Z_u_list[layer] = (torch.spmm(self.sparse_dropout(), self.E_i_list[layer-1]))
            self.Z_i_list[layer] = (torch.spmm(self.sparse_dropout().transpose(0,1), self.E_u_list[layer-1]))

            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer-1]
            self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
            ut_eu = self.ut @ self.E_u_list[layer-1]
            self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

            # aggregate
            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]
            
        self.G_u = sum(self.G_u_list)
        self.G_i = sum(self.G_i_list)

        # aggregate across layers
        self.E_u = sum(self.E_u_list)
        self.E_i = sum(self.E_i_list)
        
        return self.E_u, self.E_i
    
    def cal_cl_loss(self, uids, iids):
        G_u_norm = self.G_u
        E_u_norm = self.E_u
        G_i_norm = self.G_i
        E_i_norm = self.E_i
        neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
        loss_s = -pos_score + neg_score
        return loss_s


