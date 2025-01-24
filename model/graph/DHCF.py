# =============================================================================
# # w/ FF
# import torch
# import torch.nn as nn
# from base.graph_recommender import GraphRecommender
# from util.conf import OptionConf
# from util.sampler import next_batch_pairwise
# from base.torch_interface import TorchGraphInterface
# from util.loss_torch import bpr_loss,l2_reg_loss
# import os
# import numpy as np
# import random
# import scipy as sp
# # paper: DHCF: Dual Channel Hypergraph Collaborative Filtering. KDD'20
# # No official open-source given
# # We follow the implementation of 'QRec', which only uses 1-hop hypergraph in case of over-smoothing
# 
# seed = 0
# np.random.seed(seed)
# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.manual_seed(seed)
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False
# 
# class DHCF(GraphRecommender):
#     def __init__(self, conf, training_set, test_set):
#         super(DHCF, self).__init__(conf, training_set, test_set)
#         args = OptionConf(self.config['DHCF'])
#         self.n_layers = int(args['-n_layer'])
#         self.model = DHCF_Encoder(self.data, self.emb_size, self.n_layers)
#         
#         self.binary = self.data.interaction_mat.tocoo()
#         self.binary_row = self.binary.row
#         self.binary_col = self.binary.col
#         self.binary_data = self.binary.data
#         
#         self.binary_row_t = torch.tensor(self.binary_row, dtype=torch.int32)
#         self.binary_col_t = torch.tensor(self.binary_col, dtype=torch.int32)
#         self.binary_data_t = torch.tensor(self.binary_data, dtype=torch.bool)
#         self.observed_adj = torch.sparse_coo_tensor(torch.stack([self.binary_row_t, self.binary_col_t]), self.binary_data_t, (self.data.user_num, self.data.item_num), dtype=torch.bool)
#         self.unobserved_adj = ~(self.observed_adj.to_dense()).cuda()
#         
#     def train(self):
#         model = self.model.cuda()
#         all_performances = []
#         for self.current_layer in range(self.n_layers):
#             early_stopping = False
#             epoch = 0
#             
#             # Refresh the best performance
#             self.bestPerformance = []
#             
#             # Layer normalization
#             if self.current_layer != 0:
#                 #model.embedding_dict['user_emb'] = nn.functional.normalize(model.embedding_dict['user_emb'], p=2, dim=1)
#                 #model.embedding_dict['item_emb'] = nn.functional.normalize(model.embedding_dict['item_emb'], p=2, dim=1)
#                 model.embedding_dict['user_emb'] = nn.functional.normalize(model.tmp_u, p=2, dim=1)
#                 model.embedding_dict['item_emb'] = nn.functional.normalize(model.tmp_i, p=2, dim=1)
#             optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
#             while not early_stopping:
#                 for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
#                     user_idx, pos_idx, neg_idx = batch
#                     rec_user_emb, rec_item_emb = model(k=self.current_layer)
#                     
#                     # negative sample generation w/ neg_factor - memory saved, smaller interacted mat
#                     # mask = np.isin(self.binary_row, [user_idx])
#                     # index_dict = {value: index for index, value in enumerate(user_idx)}
#                     # mapped_row = [index_dict[element] for element in self.binary_row[mask]]
#                     # i = torch.LongTensor(np.array([mapped_row, self.binary_col[mask]]))
#                     # v = torch.FloatTensor(self.binary_data[mask])
#                     # self.unobserved_adj = torch.logical_not(torch.sparse_coo_tensor(i, v, [len(user_idx), self.data.item_num]).cuda().to_dense().to(torch.bool)).to(torch.float32)    
#                     
#                     # ui_score = torch.matmul(rec_user_emb[user_idx], rec_item_emb.transpose(0, 1))
#                     # _, indices = torch.sort(ui_score, dim=1, descending=True, stable=True) # for stable sampling - 1-1
#                     # # _, indices = torch.topk(ui_score, int(self.neg_factor*self.data.item_num*1), dim=1) # for faster sampling - 1-2
#                     # neg_idx = torch.zeros_like(ui_score, dtype=torch.float32)
#                     # # neg_idx.scatter_(1, indices[:,:int(self.neg_factor*self.data.item_num)], 1.0) # for stable sampling - 2-1
#                     # neg_idx.scatter_(1, indices, 1.0) # for faster sampling - 2-2
#                     # neg_idx = torch.logical_and(neg_idx, self.unobserved_adj[user_idx]).to(torch.half)
#                     # neg_idx = torch.multinomial(neg_idx+1e-4, 1).squeeze(1)
#                     
#                     # negative sampling generation w/ neg_factor - time saved
#                     ui_score = torch.matmul(rec_user_emb[user_idx], rec_item_emb.transpose(0, 1))
#                     # _, indices = torch.sort(ui_score, dim=1, descending=True, stable=True) # for stable sampling - 1-1
#                     _, indices = torch.topk(ui_score, int(self.neg_factor*self.data.item_num*1), dim=1) # for faster sampling - 1-2
#                     neg_idx = torch.zeros_like(ui_score, dtype=torch.bool)
#                     # neg_idx.scatter_(1, indices[:,:int(self.neg_factor*self.data.item_num)], True) # for stable sampling - 2-1
#                     neg_idx.scatter_(1, indices, 1.0) # for faster sampling - 2-2
#                     neg_idx = torch.logical_and(neg_idx, self.unobserved_adj[user_idx]).to(torch.half)
#                     neg_idx = torch.multinomial(neg_idx+1e-4, 1).squeeze(1)
#                     
#                     user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
#                     batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
#                     # Backward and optimize
#                     optimizer.zero_grad()
#                     batch_loss.backward()
#                     optimizer.step()
#                     if n % 100==0 and n>0:
#                         print('layer:', self.current_layer, 'training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
#                 with torch.no_grad():
#                     print("\nLayer %d:"%self.current_layer)
#                     self.user_emb, self.item_emb = model(k=self.current_layer, training=False)
#                 _, early_stopping = self.fast_evaluation(epoch)
#                 epoch += 1
#                 
#             all_performances.append(self.bestPerformance[1])
#         self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
#         with open('performance.txt','a') as fp:
#             for n, performance in enumerate(all_performances):
#                 fp.write("At layer %d"%n + str(performance)+"\n")
#     
#     def save(self):
#         with torch.no_grad():
#             self.best_user_emb, self.best_item_emb = self.model.forward(k=self.current_layer, training=False)
# 
#     def predict(self, u):
#         u = self.data.get_user_id(u)
#         score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
#         return score.cpu().numpy()
# 
# class DHCF_Encoder(nn.Module):
#     def __init__(self, data, emb_size, n_layers):
#         super(DHCF_Encoder, self).__init__()
#         self.data = data
#         self.latent_size = emb_size
#         self.layers = n_layers
#         self.norm_inter = data.norm_inter
#         self.embedding_dict, self.weight_dict = self._init_model()
#         self.H_u, self.H_i = self.build_hypergraph(self.norm_inter)
#         self.H_u = TorchGraphInterface.convert_sparse_mat_to_tensor(self.H_u).cuda()
#         self.H_i = TorchGraphInterface.convert_sparse_mat_to_tensor(self.H_i).cuda()
#     
#     def _init_model(self):
#         initializer = nn.init.xavier_uniform_
#         embedding_dict = nn.ParameterDict({
#             'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
#             'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
#         })
#         weight_dict = {}
#         for k in range(self.layers):
#             weight_dict['layer%d'%k] = nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size)))
#         weight_dict = nn.ParameterDict(weight_dict)
#         return embedding_dict, weight_dict
#         
#     def build_hypergraph(self, A):
#         #Build incidence matrix
#         #H_u = sp.sparse.hstack([A,A.dot(A.transpose().dot(A))])
#         H_u = A
#         D_u_v = H_u.sum(axis=1).reshape(1,-1)
#         D_u_e = H_u.sum(axis=0).reshape(1,-1)
#         temp1 = (H_u.transpose().multiply(np.sqrt(1.0/D_u_v))).transpose()
#         temp2 = temp1.transpose()
#         A_u = temp1.multiply(1.0/D_u_e).dot(temp2)
#         A_u = A_u.tocoo()
#         
#         #H_i = sp.sparse.hstack([A.transpose(),A.transpose().dot(A.dot(A.transpose()))])
#         H_i = A.transpose()
#         D_i_v = H_i.sum(axis=1).reshape(1,-1)
#         D_i_e = H_i.sum(axis=0).reshape(1,-1)
#         temp1 = (H_i.transpose().multiply(np.sqrt(1.0 / D_i_v))).transpose()
#         temp2 = temp1.transpose()
#         A_i = temp1.multiply(1.0 / D_i_e).dot(temp2)
#         A_i = A_i.tocoo()
#         return A_u, A_i
# 
#     def forward(self, k, training=True):
#         user_embeddings = self.embedding_dict['user_emb']
#         item_embeddings = self.embedding_dict['item_emb']
#         all_user_embeddings = [user_embeddings]
#         all_item_embeddings = [item_embeddings]
# 
#         def w_training(emb):
#             return nn.Dropout(p=0.1)(emb)
# 
#         def wo_training(emb):
#             return emb
#         
#         new_user_embeddings = torch.sparse.mm(self.H_u, user_embeddings)
#         new_item_embeddings = torch.sparse.mm(self.H_i, item_embeddings)
#         new_user_embeddings = torch.matmul(new_user_embeddings, self.weight_dict['layer%d'%k]) + new_user_embeddings
#         new_item_embeddings = torch.matmul(new_item_embeddings, self.weight_dict['layer%d'%k]) + new_item_embeddings
#         new_user_embeddings = nn.LeakyReLU()(new_user_embeddings)
#         new_item_embeddings = nn.LeakyReLU()(new_item_embeddings)
#         if training:
#             new_user_embeddings = w_training(new_user_embeddings)
#             new_item_embeddings = w_training(new_item_embeddings)
#         else:
#             new_user_embeddings = wo_training(new_user_embeddings)
#             new_item_embeddings = wo_training(new_item_embeddings) 
#         new_user_embeddings = nn.functional.normalize(new_user_embeddings, p=2, dim=1)
#         new_item_embeddings = nn.functional.normalize(new_item_embeddings, p=2, dim=1)
#         self.tmp_u = new_user_embeddings.clone().detach() # for next layer
#         self.tmp_i = new_item_embeddings.clone().detach() # for next layer
#         all_user_embeddings += [new_user_embeddings]
#         all_item_embeddings += [new_item_embeddings]
#         final_user_embeddings = torch.concat(all_user_embeddings, dim=1)
#         final_item_embeddings = torch.concat(all_item_embeddings, dim=1)
#         return final_user_embeddings, final_item_embeddings #new_user_embeddings, new_item_embeddings
#     
# =============================================================================

# Original Code
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
# paper: DHCF: Dual Channel Hypergraph Collaborative Filtering. KDD'20
# No official open-source given
# We follow the implementation of 'QRec', which only uses 1-hop hypergraph in case of over-smoothing

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

class DHCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(DHCF, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['DHCF'])
        self.n_layers = int(args['-n_layer'])
        self.model = DHCF_Encoder(self.data, self.emb_size, self.n_layers)
        
        self.binary = self.data.interaction_mat.tocoo()
        self.binary_row = self.binary.row
        self.binary_col = self.binary.col
        self.binary_data = self.binary.data
        
        self.binary_row_t = torch.tensor(self.binary_row, dtype=torch.int32)
        self.binary_col_t = torch.tensor(self.binary_col, dtype=torch.int32)
        self.binary_data_t = torch.tensor(self.binary_data, dtype=torch.bool)
        self.observed_adj = torch.sparse_coo_tensor(torch.stack([self.binary_row_t, self.binary_col_t]), self.binary_data_t, (self.data.user_num, self.data.item_num), dtype=torch.bool)
        self.unobserved_adj = ~(self.observed_adj.to_dense()).cuda()
        
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        while not early_stopping:
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                
            	# # negative sample generation w/ neg_factor - memory saved, smaller interacted mat
             #    # mask = np.isin(self.binary_row, [user_idx])
             #    # index_dict = {value: index for index, value in enumerate(user_idx)}
             #    # mapped_row = [index_dict[element] for element in self.binary_row[mask]]
             #    # i = torch.LongTensor(np.array([mapped_row, self.binary_col[mask]]))
             #    # v = torch.FloatTensor(self.binary_data[mask])
             #    # self.unobserved_adj = torch.logical_not(torch.sparse_coo_tensor(i, v, [len(user_idx), self.data.item_num]).cuda().to_dense().to(torch.bool)).to(torch.float32)    
                
             #    # ui_score = torch.matmul(rec_user_emb[user_idx], rec_item_emb.transpose(0, 1))
             #    # _, indices = torch.sort(ui_score, dim=1, descending=True, stable=True) # for stable sampling - 1-1
             #    # # _, indices = torch.topk(ui_score, int(self.neg_factor*self.data.item_num*1), dim=1) # for faster sampling - 1-2
             #    # neg_idx = torch.zeros_like(ui_score, dtype=torch.float32)
             #    # # neg_idx.scatter_(1, indices[:,:int(self.neg_factor*self.data.item_num)], 1.0) # for stable sampling - 2-1
             #    # neg_idx.scatter_(1, indices, 1.0) # for faster sampling - 2-2
             #    # neg_idx = torch.logical_and(neg_idx, self.unobserved_adj[user_idx]).to(torch.half)
             #    # neg_idx = torch.multinomial(neg_idx+1e-4, 1).squeeze(1)
                
             #    # negative sampling generation w/ neg_factor - time saved
             #    ui_score = torch.matmul(rec_user_emb[user_idx], rec_item_emb.transpose(0, 1))
             #    # _, indices = torch.sort(ui_score, dim=1, descending=True, stable=True) # for stable sampling - 1-1
             #    _, indices = torch.topk(ui_score, int(self.neg_factor*self.data.item_num*1), dim=1) # for faster sampling - 1-2
             #    neg_idx = torch.zeros_like(ui_score, dtype=torch.bool)
             #    # neg_idx.scatter_(1, indices[:,:int(self.neg_factor*self.data.item_num)], True) # for stable sampling - 2-1
             #    neg_idx.scatter_(1, indices, 1.0) # for faster sampling - 2-2
             #    neg_idx = torch.logical_and(neg_idx, self.unobserved_adj[user_idx]).to(torch.half)
             #    neg_idx = torch.multinomial(neg_idx+1e-4, 1).squeeze(1)
                
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model(training=False)
            _, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")
    
    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward(training=False)

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class DHCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(DHCF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_inter = data.norm_inter
        self.embedding_dict, self.weight_dict = self._init_model()
        self.H_u, self.H_i = self.build_hypergraph(self.norm_inter)
        self.H_u = TorchGraphInterface.convert_sparse_mat_to_tensor(self.H_u).cuda()
        self.H_i = TorchGraphInterface.convert_sparse_mat_to_tensor(self.H_i).cuda()
    
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        weight_dict = {}
        for k in range(self.layers):
            weight_dict['layer%d'%k] = nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size)))
        weight_dict = nn.ParameterDict(weight_dict)
        return embedding_dict, weight_dict
        
    def build_hypergraph(self, A):
        #Build incidence matrix
        #H_u = sp.sparse.hstack([A,A.dot(A.transpose().dot(A))])
        H_u = A
        D_u_v = H_u.sum(axis=1).reshape(1,-1)
        D_u_e = H_u.sum(axis=0).reshape(1,-1)
        temp1 = (H_u.transpose().multiply(np.sqrt(1.0/D_u_v))).transpose()
        temp2 = temp1.transpose()
        A_u = temp1.multiply(1.0/D_u_e).dot(temp2)
        A_u = A_u.tocoo()
        
        #H_i = sp.sparse.hstack([A.transpose(),A.transpose().dot(A.dot(A.transpose()))])
        H_i = A.transpose()
        D_i_v = H_i.sum(axis=1).reshape(1,-1)
        D_i_e = H_i.sum(axis=0).reshape(1,-1)
        temp1 = (H_i.transpose().multiply(np.sqrt(1.0 / D_i_v))).transpose()
        temp2 = temp1.transpose()
        A_i = temp1.multiply(1.0 / D_i_e).dot(temp2)
        A_i = A_i.tocoo()
        return A_u, A_i

    def forward(self, training=True):
        user_embeddings = self.embedding_dict['user_emb']
        item_embeddings = self.embedding_dict['item_emb']
        all_user_embeddings = [user_embeddings]
        all_item_embeddings = [item_embeddings]

        def w_training(emb):
            return nn.Dropout(p=0.1)(emb)

        def wo_training(emb):
            return emb
        
        for k in range(self.layers):
            new_user_embeddings = torch.sparse.mm(self.H_u, all_user_embeddings[k])
            new_item_embeddings = torch.sparse.mm(self.H_i, all_item_embeddings[k])

            new_user_embeddings = torch.matmul(new_user_embeddings, self.weight_dict['layer%d'%k]) + new_user_embeddings
            new_item_embeddings = torch.matmul(new_item_embeddings, self.weight_dict['layer%d'%k]) + new_item_embeddings
            new_user_embeddings = nn.LeakyReLU()(new_user_embeddings)
            new_item_embeddings = nn.LeakyReLU()(new_item_embeddings)
            if training:
                new_user_embeddings = w_training(new_user_embeddings)
                new_item_embeddings = w_training(new_item_embeddings)
            else:
                new_user_embeddings = wo_training(new_user_embeddings)
                new_item_embeddings = wo_training(new_item_embeddings)
            new_user_embeddings = nn.functional.normalize(new_user_embeddings, p=2, dim=1)
            new_item_embeddings = nn.functional.normalize(new_item_embeddings, p=2, dim=1)
            all_user_embeddings += [new_user_embeddings]
            all_item_embeddings += [new_item_embeddings]
        final_user_embeddings = torch.concat(all_user_embeddings, dim=1)
        final_item_embeddings = torch.concat(all_item_embeddings, dim=1)
        # final_user_embeddings = all_user_embeddings[-1]
        # final_item_embeddings = all_item_embeddings[-1]
        return final_user_embeddings, final_item_embeddings




