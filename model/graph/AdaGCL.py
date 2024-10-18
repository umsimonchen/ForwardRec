import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss, InfoNCE
import os
import numpy as np
import random
import pickle
# paper: AdaGCL: Adaptive Graph Contrastive Learning for Recommendation, KDD'23

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

class AdaGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(AdaGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['AdaGCL'])
        self.n_layers = int(args['-n_layer'])
        self.tau = float(args['-tau'])
        self.ssl_loss = float(args['-ssl_loss'])
        self.model = AdaGCL_Encoder(self.data, self.emb_size, self.n_layers, self.tau, self.ssl_loss)
        self.binary = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda().to_dense()
        
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
                
                rec_user_emb, rec_item_emb, gen_user_emb, gen_item_emb, den_user_emb, den_item_emb, mean, std, gen_graph, mask_graph= model()
                
                ui_score = torch.matmul(rec_user_emb[user_idx], rec_item_emb.transpose(0, 1))
                ui_score = torch.clamp(ui_score, min=0.0)
                neg_idx = torch.multinomial(ui_score, 1, replacement=False).squeeze(1)
                
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                gen_user_emb, gen_pos_item_emb, gen_neg_item_emb = gen_user_emb[user_idx], gen_item_emb[pos_idx], gen_item_emb[neg_idx]
                den_user_emb, den_pos_item_emb, den_neg_item_emb = den_user_emb[user_idx], den_item_emb[pos_idx], den_item_emb[neg_idx]
                
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                gen_batch_loss = bpr_loss(gen_user_emb, gen_pos_item_emb, gen_neg_item_emb) + l2_reg_loss(self.reg, gen_user_emb,gen_pos_item_emb,gen_neg_item_emb)/self.batch_size
                den_batch_loss = bpr_loss(den_user_emb, den_pos_item_emb, den_neg_item_emb) + l2_reg_loss(self.reg, den_user_emb,den_pos_item_emb,den_neg_item_emb)/self.batch_size
                cl_loss = 0
                cl_loss += self.ssl_loss * model.cal_cl_loss(gen_user_emb, den_user_emb, self.tau)/self.batch_size
                cl_loss += self.ssl_loss * model.cal_cl_loss(gen_pos_item_emb, den_pos_item_emb, self.tau)/self.batch_size
                
                kl_loss = - 0.5 * (1 + 2 * torch.log(std+1e-5) - mean**2 - std**2).sum()/(self.data.user_num*self.data.item_num)
                
                gen_graph = torch.clamp(gen_graph, min=1e-5, max=1.0-1e-5)
                dis_loss = -(torch.mul(torch.log(gen_graph[user_idx]), self.binary[user_idx]) + torch.mul(torch.log(1-gen_graph[user_idx]), 1-self.binary[user_idx])).sum()/(self.data.user_num*self.data.item_num)
                
                mask_loss = 0
                #for k in range(self.n_layers):
                mask_loss += torch.sigmoid(mask_graph[user_idx]).sum() / (self.data.user_num*self.data.item_num)
                
                total_loss = batch_loss + gen_batch_loss + den_batch_loss + cl_loss + kl_loss + dis_loss + mask_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb, _, _, _, _, _, _, _, _ = model()
            measure, early_stopping = self.fast_evaluation(epoch)
            record_list.append(measure)
            loss_list.append(batch_loss.item())
            epoch += 1
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")
        # record training loss        
        with open('training_record','wb') as fp:
            pickle.dump([record_list, loss_list], fp)

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, _, _, _, _, _, _, _, _ = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class AdaGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, tau, ssl_loss):
        super(AdaGCL_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.tau = tau
        self.ssl_loss = ssl_loss
        self.norm_adj = data.norm_adj
        self.embedding_dict, self.mlp_dict, self.mask_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.dense_norm_adj = self.sparse_norm_adj.to_dense()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        mlp_dict = nn.ParameterDict({
             'mean': nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size))),
             'std': nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size))),
             'decoder': nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size))),
        })
        mask_dict = {}
        for k in range(self.layers):
            mask_dict['layer%d'%k] = nn.Parameter(initializer(torch.empty(1, self.latent_size)))
        mask_dict = nn.ParameterDict(mask_dict)
        return embedding_dict, mlp_dict, mask_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        den_all_embeddings = [ego_embeddings]
        # main task 
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings + all_embeddings[-1]]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        
        # generator
        mean = torch.relu(torch.matmul(all_embeddings, self.mlp_dict['mean']))
        std = torch.relu(torch.matmul(all_embeddings, self.mlp_dict['std']))
        x = torch.randn(mean.shape).cuda() * std + mean
        
        # decode
        decoded_embeddings = torch.relu(torch.matmul(x, self.mlp_dict['decoder']))
        gen_user_all_embeddings = [self.embedding_dict['user_emb']]
        gen_item_all_embeddings = [self.embedding_dict['item_emb']]
        gen_graph = torch.matmul(decoded_embeddings[:self.data.user_num], decoded_embeddings[self.data.user_num:].transpose(0, 1))
        gen_graph_u = torch.nn.functional.normalize(gen_graph, p=1)
        gen_graph_i = torch.nn.functional.normalize(gen_graph.transpose(0, 1), p=1)
        for k in range(self.layers):
            gen_user_all_embeddings += [torch.matmul(gen_graph_u, gen_item_all_embeddings[-1])]
            gen_item_all_embeddings += [torch.matmul(gen_graph_i, gen_user_all_embeddings[-1])]
        gen_user_all_embeddings = torch.stack(gen_user_all_embeddings, dim=1)
        gen_user_all_embeddings = torch.mean(gen_user_all_embeddings, dim=1)
        gen_item_all_embeddings = torch.stack(gen_item_all_embeddings, dim=1)
        gen_item_all_embeddings = torch.mean(gen_item_all_embeddings, dim=1)
        
        # denoise
        mask_graph = 0
        for k in range(self.layers):
            den_ego_embeddings = torch.sigmoid(torch.mul(self.mask_dict['layer%d'%k], den_all_embeddings[-1]))
            den_ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, den_ego_embeddings)
            den_all_embeddings += [den_ego_embeddings]
            mask_graph += torch.sigmoid(torch.matmul(den_ego_embeddings[:self.data.user_num], den_ego_embeddings[self.data.user_num:].transpose(0, 1)))
        den_all_embeddings = torch.stack(den_all_embeddings, dim=1)
        den_all_embeddings = torch.mean(den_all_embeddings, dim=1)
        den_user_all_embeddings = den_all_embeddings[:self.data.user_num]
        den_item_all_embeddings = den_all_embeddings[self.data.user_num:]
        
        return user_all_embeddings, item_all_embeddings, gen_user_all_embeddings, gen_item_all_embeddings, den_user_all_embeddings, den_item_all_embeddings, mean, std, gen_graph, mask_graph
    
    def cal_cl_loss(self, view1, view2, temp):
        # user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        #return user_cl_loss + item_cl_loss
        return InfoNCE(view1,view2,temp)




# =============================================================================
# import torch
# import torch.nn as nn
# from base.graph_recommender import GraphRecommender
# from util.conf import OptionConf
# from util.sampler import next_batch_pairwise
# from base.torch_interface import TorchGraphInterface
# from util.loss_torch import bpr_loss,l2_reg_loss, InfoNCE
# import os
# import numpy as np
# import random
# import pickle
# # paper: AdaGCL: Adaptive Graph Contrastive Learning for Recommendation, KDD'23
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
# class AdaGCL(GraphRecommender):
#     def __init__(self, conf, training_set, test_set):
#         super(AdaGCL, self).__init__(conf, training_set, test_set)
#         args = OptionConf(self.config['AdaGCL'])
#         self.n_layers = int(args['-n_layer'])
#         self.tau = float(args['-tau'])
#         self.ssl_loss = float(args['-ssl_loss'])
#         self.model = AdaGCL_Encoder(self.data, self.emb_size, self.n_layers, self.tau, self.ssl_loss)
#         self.binary = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda().to_dense()
#         
#     def train(self):
#         record_list = []
#         loss_list = []
#         model = self.model.cuda()
#         optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
#         early_stopping = False
#         epoch = 0
#         while not early_stopping:
#             for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
#                 user_idx, pos_idx, neg_idx = batch
#                 rec_user_emb, rec_item_emb, gen_user_emb, gen_item_emb, den_user_emb, den_item_emb, mean, std = model()
#                 
#                 ui_score = torch.matmul(rec_user_emb[user_idx], rec_item_emb.transpose(0, 1))
#                 ui_score = torch.clamp(ui_score, min=0.0)
#                 neg_idx = torch.multinomial(ui_score, 1, replacement=False).squeeze(1)
#                 
#                 user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
#                 gen_user_emb, gen_pos_item_emb, gen_neg_item_emb = gen_user_emb[user_idx], gen_item_emb[pos_idx], gen_item_emb[neg_idx]
#                 den_user_emb, den_pos_item_emb, den_neg_item_emb = den_user_emb[user_idx], den_item_emb[pos_idx], den_item_emb[neg_idx]
#                 
#                 batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
#                 gen_batch_loss = bpr_loss(gen_user_emb, gen_pos_item_emb, gen_neg_item_emb) + l2_reg_loss(self.reg, gen_user_emb,gen_pos_item_emb,gen_neg_item_emb)/self.batch_size
#                 den_batch_loss = bpr_loss(den_user_emb, den_pos_item_emb, den_neg_item_emb) + l2_reg_loss(self.reg, den_user_emb,den_pos_item_emb,den_neg_item_emb)/self.batch_size
#                 cl_loss = 0
#                 cl_loss += self.ssl_loss * model.cal_cl_loss(gen_user_emb, den_user_emb, self.tau)/self.batch_size
#                 cl_loss += self.ssl_loss * model.cal_cl_loss(gen_pos_item_emb, den_pos_item_emb, self.tau)/self.batch_size
#                 
#                 kl_loss = - 0.5 * (1 + 2 * torch.log(std+1e-5) - mean**2 - std**2).sum()/(self.data.user_num*self.data.item_num)
# 
#                 # generate graph has limited gain
#                 # gen_graph = torch.matmul(gen_user_emb[user_idx], gen_item_emb.transpose(0, 1))
#                 # gen_graph = torch.sigmoid(gen_graph)
#                 # dis_loss = -(torch.mul(torch.log(gen_graph+1e-5), self.binary[user_idx]) + torch.mul(torch.log(1-gen_graph+1e-5), 1-self.binary[user_idx])).sum()/(self.data.user_num*self.data.item_num)
# 
#                 # mask graph has limited gain
#                 # mask_loss = 0
#                 # for k in range(self.n_layers):
#                 # mask_graph = torch.matmul(den_user_emb[user_idx], den_item_emb.transpose(0, 1))
#                 # mask_loss += torch.sigmoid(mask_graph[user_idx]).sum() / (self.data.user_num*self.data.item_num)
#                 
#                 total_loss = batch_loss + gen_batch_loss + den_batch_loss + cl_loss + kl_loss #+ dis_loss + mask_loss 
#                 
#                 # Backward and optimize
#                 optimizer.zero_grad()
#                 total_loss.backward()
#                 optimizer.step()
#                 if n % 100==0 and n>0:
#                     print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
#             with torch.no_grad():
#                 self.user_emb, self.item_emb, _, _, _, _, _, _ = model()
#             measure, early_stopping = self.fast_evaluation(epoch)
#             record_list.append(measure)
#             loss_list.append(batch_loss.item())
#             epoch += 1
#         self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
#         with open('performance.txt','a') as fp:
#             fp.write(str(self.bestPerformance[1])+"\n")
#         # record training loss        
#         with open('training_record','wb') as fp:
#             pickle.dump([record_list, loss_list], fp)
# 
#     def save(self):
#         with torch.no_grad():
#             self.best_user_emb, self.best_item_emb, _, _, _, _, _, _ = self.model.forward()
# 
#     def predict(self, u):
#         u = self.data.get_user_id(u)
#         score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
#         return score.cpu().numpy()
# 
# class AdaGCL_Encoder(nn.Module):
#     def __init__(self, data, emb_size, n_layers, tau, ssl_loss):
#         super(AdaGCL_Encoder, self).__init__()
#         self.data = data
#         self.latent_size = emb_size
#         self.layers = n_layers
#         self.tau = tau
#         self.ssl_loss = ssl_loss
#         self.norm_adj = data.norm_adj
#         self.embedding_dict, self.mlp_dict, self.mask_dict = self._init_model()
#         self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
#         self.dense_norm_adj = self.sparse_norm_adj.to_dense()
# 
#     def _init_model(self):
#         initializer = nn.init.xavier_uniform_
#         embedding_dict = nn.ParameterDict({
#             'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
#             'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
#         })
#         mlp_dict = nn.ParameterDict({
#              'mean': nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size))),
#              'std': nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size))),
#              'decoder': nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size))),
#         })
#         mask_dict = {}
#         for k in range(self.layers):
#             mask_dict['layer%d'%k] = nn.Parameter(initializer(torch.empty(1, self.latent_size)))
#         mask_dict = nn.ParameterDict(mask_dict)
#         return embedding_dict, mlp_dict, mask_dict
# 
#     def forward(self):
#         ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
#         all_embeddings = [ego_embeddings]
#         den_all_embeddings = [ego_embeddings]
#         # main task 
#         for k in range(self.layers):
#             ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
#             all_embeddings += [ego_embeddings + all_embeddings[-1]]
#         all_embeddings = torch.stack(all_embeddings, dim=1)
#         all_embeddings = torch.mean(all_embeddings, dim=1)
#         user_all_embeddings = all_embeddings[:self.data.user_num]
#         item_all_embeddings = all_embeddings[self.data.user_num:]
#         
#         # generator
#         mean = torch.relu(torch.matmul(all_embeddings, self.mlp_dict['mean']))
#         std = torch.relu(torch.matmul(all_embeddings, self.mlp_dict['std']))
#         x = torch.randn(mean.shape).cuda() * std + mean
#         
#         # decode
#         decoded_embeddings = torch.relu(torch.matmul(x, self.mlp_dict['decoder']))
#         gen_user_all_embeddings = [self.embedding_dict['user_emb']]
#         gen_item_all_embeddings = [self.embedding_dict['item_emb']]
#         for k in range(self.layers):
#             gen_user_all_embeddings += [torch.matmul(decoded_embeddings[:self.data.user_num], torch.matmul(decoded_embeddings[self.data.user_num:].transpose(0, 1), gen_item_all_embeddings[-1]))]
#             gen_item_all_embeddings += [torch.matmul(decoded_embeddings[self.data.user_num:], torch.matmul(decoded_embeddings[:self.data.user_num].transpose(0, 1), gen_user_all_embeddings[-1]))]
#         gen_user_all_embeddings = torch.stack(gen_user_all_embeddings, dim=1)
#         gen_user_all_embeddings = torch.mean(gen_user_all_embeddings, dim=1)
#         gen_item_all_embeddings = torch.stack(gen_item_all_embeddings, dim=1)
#         gen_item_all_embeddings = torch.mean(gen_item_all_embeddings, dim=1)
#         
#         # denoise
#         for k in range(self.layers):
#             den_ego_embeddings = torch.sigmoid(torch.mul(self.mask_dict['layer%d'%k], den_all_embeddings[-1]))
#             den_ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, den_ego_embeddings)
#             den_all_embeddings += [den_ego_embeddings]
#         den_all_embeddings = torch.stack(den_all_embeddings, dim=1)
#         den_all_embeddings = torch.mean(den_all_embeddings, dim=1)
#         den_user_all_embeddings = den_all_embeddings[:self.data.user_num]
#         den_item_all_embeddings = den_all_embeddings[self.data.user_num:]
#         
#         return user_all_embeddings, item_all_embeddings, gen_user_all_embeddings, gen_item_all_embeddings, den_user_all_embeddings, den_item_all_embeddings, mean, std
#     
#     def cal_cl_loss(self, view1, view2, temp):
#         # user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
#         # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
#         #return user_cl_loss + item_cl_loss
#         return InfoNCE(view1,view2,temp)
#
# =============================================================================

