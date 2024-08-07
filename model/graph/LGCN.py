# =============================================================================
# # # Original Code
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
# import scipy.sparse.linalg
# import pickle
# # paper: LGCN: Low-Pass Graph Convolutional Network for Recommendation. AAAI'22
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
# class LGCN(GraphRecommender):
#     def __init__(self, conf, training_set, test_set):
#         super(LGCN, self).__init__(conf, training_set, test_set)
#         args = OptionConf(self.config['LGCN'])
#         self.n_layers = int(args['-n_layer'])
#         self.frequency = int(args['-frequency'])
#         self.unobserved_adj = torch.logical_not(TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda().to_dense().to(torch.bool)).to(torch.float32)
#         self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers, self.frequency)
#         
#     def train(self):
#         model = self.model.cuda()
#         optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
#         early_stopping = False
#         epoch = 0
#         while not early_stopping:
#             for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
#                 user_idx, pos_idx, neg_idx = batch
#                 rec_user_emb, rec_item_emb = model()
#                 
#                 # # w/ NS
#                 # ui_score = torch.matmul(rec_user_emb[user_idx], rec_item_emb.transpose(0, 1))
#                 # _, indices = torch.sort(ui_score, dim=1, descending=True, stable=True)
#                 # neg_idx = torch.zeros_like(ui_score, dtype=torch.float32)
#                 # neg_idx.scatter_(1, indices[:,:int(0.06*self.data.item_num)], 1.0)
#                 # neg_idx = torch.mul(neg_idx, self.unobserved_adj[user_idx])
#                 # _, neg_idx = torch.sort(torch.mul(torch.randn_like(neg_idx), neg_idx), dim=1, descending=True, stable=True)
#                 # neg_idx = neg_idx[:,0]
#                 
#                 user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
#                 batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
#                 # Backward and optimize
#                 optimizer.zero_grad()
#                 batch_loss.backward()
#                 optimizer.step()
#                 if n % 100==0 and n>0:
#                     print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
#             with torch.no_grad():
#                 self.user_emb, self.item_emb = model()
#             _, early_stopping = self.fast_evaluation(epoch)
#             epoch += 1
#         self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
#         with open('performance.txt','a') as fp:
#             fp.write(str(self.bestPerformance[1])+"\n")
# 
#     def save(self):
#         with torch.no_grad():
#             self.best_user_emb, self.best_item_emb = self.model.forward()
# 
#     def predict(self, u):
#         u = self.data.get_user_id(u)
#         score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
#         return score.cpu().numpy()
# 
# class LGCN_Encoder(nn.Module):
#     def __init__(self, data, emb_size, n_layers, frequency):
#         super(LGCN_Encoder, self).__init__()
#         self.data = data
#         self.latent_size = emb_size
#         self.layers = n_layers
#         self.frequency = frequency
#         self.norm_adj = data.norm_adj
#         # with open('amazon-beauty', 'wb') as fp:
#         #     pickle.dump(self.norm_adj, fp)
#         with open('dataset/yelp2018/eigen','rb') as fp:
#             [self.e, self.v] = pickle.load(fp)
#         self.e = torch.tensor(self.e).cuda().float()
#         self.v = torch.tensor(self.v).cuda().float()
#         self.embedding_dict, self.filter_dict = self._init_model()
# 
#     def _init_model(self):
#         initializer = nn.init.xavier_uniform_
#         embedding_dict = nn.ParameterDict({
#             'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
#             'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
#         })
#         filter_dict = {}
#         for k in range(self.layers):
#             filter_dict['layer%d'%k] = nn.Parameter(self.e.clone())
#         filter_dict = nn.ParameterDict(filter_dict)
#         return embedding_dict, filter_dict
# 
#     def forward(self):
#         ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
#         all_embeddings = [ego_embeddings]
#         for k in range(self.layers):
#             ego_embeddings = torch.matmul(torch.matmul(self.v, torch.diag(self.filter_dict['layer%d'%k])), torch.matmul(self.v.transpose(0,1), ego_embeddings))
#             all_embeddings += [ego_embeddings]
#         all_embeddings = torch.stack(all_embeddings, dim=1)
#         all_embeddings = torch.mean(all_embeddings, dim=1)
#         user_all_embeddings = all_embeddings[:self.data.user_num]
#         item_all_embeddings = all_embeddings[self.data.user_num:]
#         return user_all_embeddings, item_all_embeddings
# =============================================================================

# FF
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
import scipy.sparse.linalg
import pickle
# paper: LGCN: Low-Pass Graph Convolutional Network for Recommendation. AAAI'22

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

class LGCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LGCN, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['LGCN'])
        self.n_layers = int(args['-n_layer'])
        self.frequency = int(args['-frequency'])
        self.unobserved_adj = torch.logical_not(TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda().to_dense().to(torch.bool)).to(torch.float32)
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers, self.frequency)
        

    def train(self):
        model = self.model.cuda()
        all_performances = []
        for self.current_layer in range(self.n_layers):
            early_stopping = False
            epoch = 0  
            # Refresh the best performance
            self.bestPerformance = []
          
            # Layer normalization
            if self.current_layer != 0:
                model.embedding_dict['user_emb'] = nn.functional.normalize(model.embedding_dict['user_emb'], p=2, dim=1)
                model.embedding_dict['item_emb'] = nn.functional.normalize(model.embedding_dict['item_emb'], p=2, dim=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
            while not early_stopping:
                for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                    user_idx, pos_idx, neg_idx = batch
                    rec_user_emb, rec_item_emb = model(self.current_layer)
                    
                    # w/ NS
                    # ui_score = torch.matmul(rec_user_emb[user_idx], rec_item_emb.transpose(0, 1))
                    # _, indices = torch.sort(ui_score, dim=1, descending=True, stable=True)
                    # neg_idx = torch.zeros_like(ui_score, dtype=torch.float32)
                    # neg_idx.scatter_(1, indices[:,:int(0.06*self.data.item_num)], 1.0)
                    # neg_idx = torch.mul(neg_idx, self.unobserved_adj[user_idx])
                    # _, neg_idx = torch.sort(torch.mul(torch.randn_like(neg_idx), neg_idx), dim=1, descending=True, stable=True)
                    # neg_idx = neg_idx[:,0]
                    
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                    batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                    # Backward and optimize
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    if n%100 == 0 and n > 0:
                        print('layer:', self.current_layer, 'training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
                    #if (self.current_layer==1 and epoch<12) or (self.current_layer==2 and epoch<16) or (self.current_layer==3 and epoch<12): #lastfm
                        #self.bestPerformance = []
                with torch.no_grad():
                    print("\nLayer %d:"%self.current_layer)
                    self.user_emb, self.item_emb = model(self.current_layer)
                _, early_stopping = self.fast_evaluation(epoch)
                epoch += 1
                
            all_performances.append(self.bestPerformance[1])
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        with open('performance.txt','a') as fp:
            for n, performance in enumerate(all_performances):
                fp.write("At layer %d"%n + str(performance)+"\n")

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward(self.current_layer)

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, frequency):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.frequency = frequency
        self.norm_adj = data.norm_adj
        with open('dataset/amazon-beauty/eigen','rb') as fp: # please read the correct eigenvalues when running this code
            [self.e, self.v] = pickle.load(fp)
        self.v = torch.tensor(self.v).cuda().float()
        self.e = torch.tensor(self.e).cuda().float()
        self.embedding_dict, self.filter_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        filter_dict = {}
        for k in range(self.layers):
            filter_dict['layer%d'%k] = nn.Parameter(self.e.clone())
        filter_dict = nn.ParameterDict(filter_dict)
        return embedding_dict, filter_dict

    def forward(self, k):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], dim=0)
        all_embeddings = [ego_embeddings]
        ego_embeddings = torch.matmul(torch.matmul(self.v, torch.diag(self.filter_dict['layer%d'%k])), torch.matmul(self.v.transpose(0,1), ego_embeddings))
        all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        # all_embeddings = all_embeddings[-1] #max
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings

