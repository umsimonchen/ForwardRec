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
# paper: Simplify to the Limit! Embedding-less Graph Collaborative Filtering for Recommender Systems. TOIS'24
# https://github.com/BlueGhostYi/ID-GRec

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

class EGCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(EGCF, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['EGCF'])
        self.n_layers = int(args['-n_layer'])
        self.tau = float(args['-tau'])
        self.lamda = float(args['-lamda'])
        self.agg_type = int(args['-agg_type'])
        self.model = EGCF_Encoder(self.data, self.emb_size, self.n_layers, self.tau, self.lamda, self.agg_type)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        while not early_stopping:
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                
                ego_pos_emb = model.embedding_dict['item_emb'][pos_idx]
                ego_neg_emb = model.embedding_dict['item_emb'][neg_idx]
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                cl_loss = self.lamda * model.cal_cl_loss(user_idx, pos_idx)
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, ego_pos_emb, ego_neg_emb) + cl_loss
                
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

class EGCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, tau, lamda, agg_type):
        super(EGCF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.tau = tau
        self.lamda = lamda
        self.agg_type = agg_type # parallel iteration==0, otherwise alternating iteration==1
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda() # (m+n)*(m+n)
        
        # constructe graph for alternating aggregation
        adjacency_matrix = data.interaction_mat
        row_sum = np.array(adjacency_matrix.sum(axis=1))
        row_d_inv = np.power(row_sum, -0.5).flatten()
        row_d_inv[np.isinf(row_d_inv)] = 0.
        row_degree_matrix = sp.diags(row_d_inv)

        col_sum = np.array(adjacency_matrix.sum(axis=0))
        col_d_inv = np.power(col_sum, -0.5).flatten()
        col_d_inv[np.isinf(col_d_inv)] = 0.
        col_degree_matrix = sp.diags(col_d_inv)

        self.norm_adj = row_degree_matrix.dot(adjacency_matrix).dot(col_degree_matrix).tocsr()
        self.sparse_norm_inter = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda() # m*n
        
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': None,
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def parallel_aggregate(self):
        item_embedding = self.embedding_dict['item_emb']
        user_embedding = nn.Tanh()(torch.sparse.mm(self.sparse_norm_inter, item_embedding))
        all_embedding = torch.cat([user_embedding, item_embedding])
        all_embeddings = []
        
        for layer in range(self.layers):
            all_embedding = nn.Tanh()(torch.sparse.mm(self.sparse_norm_adj, all_embedding))
            all_embeddings.append(all_embedding)
        final_all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.sum(final_all_embeddings, dim=1)
        
        users_emb, items_emb = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        
        return users_emb, items_emb
    
    def alternating_aggregate(self):
        item_embedding = self.embedding_dict['item_emb']
        
        all_user_embeddings = []
        all_item_embeddings = []
        
        for layer in range(self.layers):
            user_embedding = nn.Tanh()(torch.sparse.mm(self.sparse_norm_inter, item_embedding))
            item_embedding = nn.Tanh()(torch.sparse.mm(self.sparse_norm_inter.transpose(0,1), user_embedding))
            
            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)
            
        final_user_embeddings = torch.stack(all_user_embeddings, dim=1)
        final_user_embeddings = torch.sum(final_user_embeddings, dim=1)
        
        final_item_embeddings = torch.stack(all_item_embeddings, dim=1)
        final_item_embeddings = torch.sum(final_item_embeddings, dim=1)
        
        return final_user_embeddings, final_item_embeddings
        

    def forward(self):
        if self.agg_type==0:
            self.all_user_embeddings, self.all_item_embeddings = self.parallel_aggregate()
        elif self.agg_type==1:
            self.all_user_embeddings, self.all_item_embeddings = self.alternating_aggregate()
        
        return self.all_user_embeddings, self.all_item_embeddings
    
    def get_InfoNCE_loss(self, embedding_1, embedding_2, temperature):
        embedding_1 = torch.nn.functional.normalize(embedding_1)
        embedding_2 = torch.nn.functional.normalize(embedding_2)
    
        pos_score = (embedding_1 * embedding_2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
    
        ttl_score = torch.matmul(embedding_1, embedding_2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    
        cl_loss = - torch.log(pos_score / ttl_score + 10e-6)
        return torch.mean(cl_loss)
    
    def cal_cl_loss(self, user_idx, pos_idx):
        user_embedding = self.all_user_embeddings[user_idx]
        pos_embedding = self.all_item_embeddings[pos_idx]
        
        ssl_user_loss = self.get_InfoNCE_loss(user_embedding, user_embedding, self.tau)
        ssl_item_loss = self.get_InfoNCE_loss(pos_embedding, pos_embedding, self.tau)
        ssl_inter_loss = self.get_InfoNCE_loss(user_embedding, pos_embedding, self.tau)
        
        return ssl_user_loss+ssl_item_loss+ssl_inter_loss
        


