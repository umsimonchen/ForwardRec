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
# https://github.com/HKUDS/AdaGCL

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
        self.candidate_user = 100 # to save memory, only denoise for a partial of graph
        self.model = AdaGCL_Encoder(self.data, self.emb_size, self.n_layers, self.tau, self.ssl_loss, self.candidate_user)
        
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
                rec_user_emb, rec_item_emb, gen_user_emb, gen_item_emb, den_user_emb, den_item_emb, mean, std, all_denoise_graph = model()
                
                user_emb1, pos_item_emb1, neg_item_emb1 = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                user_emb2, pos_item_emb2, neg_item_emb2 = gen_user_emb[user_idx], gen_item_emb[pos_idx], gen_item_emb[neg_idx]
                user_emb3, pos_item_emb3, neg_item_emb3 = den_user_emb[user_idx], den_item_emb[pos_idx], den_item_emb[neg_idx]
                
                batch_loss = bpr_loss(user_emb1, pos_item_emb1, neg_item_emb1) + l2_reg_loss(self.reg, user_emb1, pos_item_emb1, neg_item_emb1)/self.batch_size
                gen_batch_loss = bpr_loss(user_emb2, pos_item_emb2, neg_item_emb2) + l2_reg_loss(self.reg, user_emb2, pos_item_emb2, neg_item_emb2)/self.batch_size
                den_batch_loss = bpr_loss(user_emb3, pos_item_emb3, neg_item_emb3) + l2_reg_loss(self.reg, user_emb3, pos_item_emb3, neg_item_emb3)/self.batch_size
                
                cl_loss = 0.0
                cl_loss += self.ssl_loss * model.cal_cl_loss(user_emb2, user_emb3, self.tau)/self.batch_size
                cl_loss += self.ssl_loss * model.cal_cl_loss(pos_item_emb2, pos_item_emb3, self.tau)/self.batch_size
                
                kl_loss = - 0.5 * (1 + 2 * torch.log(std+1e-5) - mean**2 - std**2).sum()/(self.data.user_num*self.data.item_num)

                # generate graph from the perspective of embedding to save memory
                dis_loss = (((rec_user_emb-gen_user_emb)**2).sum() + ((rec_item_emb-gen_item_emb)**2).sum()) / ((self.data.user_num+self.data.item_num)*self.emb_size)

                # mask graph has limited gain
                mask_loss = 0.0
                for k in range(self.n_layers):
                    mask_loss += (all_denoise_graph[k]).sum() / ((self.data.user_num+self.data.item_num)*self.candidate_user)
                
                # kl_loss, dis_loss, mask_loss destory the stability
                total_loss = batch_loss + gen_batch_loss + den_batch_loss + cl_loss #+ kl_loss + dis_loss + mask_loss 
                
                # Backward and optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb, _, _, _, _, _, _, _ = model()
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
            self.best_user_emb, self.best_item_emb, _, _, _, _, _, _, _ = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class AdaGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, tau, ssl_loss, candidate_user):
        super(AdaGCL_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.tau = tau
        self.ssl_loss = ssl_loss
        self.candidate_user = candidate_user
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
        mask_dict = nn.ParameterDict({
            'shared_layer': nn.Parameter(initializer(torch.empty(1, self.latent_size))),
        })
        return embedding_dict, mlp_dict, mask_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        gen_all_embeddings = [ego_embeddings]
        den_all_embeddings = [ego_embeddings]
        # main task 
        for k in range(self.layers):
            all_embeddings += [torch.sparse.mm(self.sparse_norm_adj, all_embeddings[-1]) + all_embeddings[-1]]
        main_all_embeddings = torch.stack(all_embeddings, dim=1)
        main_all_embeddings = torch.mean(main_all_embeddings, dim=1)
        user_all_embeddings = main_all_embeddings[:self.data.user_num]
        item_all_embeddings = main_all_embeddings[self.data.user_num:]
        
        # generator
        mean = torch.relu(torch.matmul(main_all_embeddings, self.mlp_dict['mean']))
        std = torch.relu(torch.matmul(main_all_embeddings, self.mlp_dict['std']))
        x = torch.randn(mean.shape).cuda() * std + mean
        
        # decode
        decoded_embeddings = torch.relu(torch.matmul(x, self.mlp_dict['decoder']))
        for k in range(self.layers):
            gen_all_embeddings += [torch.matmul(decoded_embeddings, torch.matmul(decoded_embeddings.transpose(0,1), gen_all_embeddings[-1]))]
        gen_all_embeddings = torch.stack(gen_all_embeddings, dim=1)
        gen_all_embeddings = torch.mean(gen_all_embeddings, dim=1)
        gen_user_all_embeddings = gen_all_embeddings[:self.data.user_num]
        gen_item_all_embeddings = gen_all_embeddings[self.data.user_num:]
        
        # denoise
        user_idx = torch.randint(0, self.data.user_num, (self.candidate_user,))
        all_denoise_graph = []
        for k in range(self.layers):
            hidden = torch.sigmoid(torch.mul(self.mask_dict['shared_layer'], all_embeddings[k]))
            denoise_graph = torch.matmul(hidden[user_idx], hidden.transpose(0,1))
            denoise_graph = torch.sigmoid(denoise_graph)
            all_denoise_graph += [denoise_graph]
            
            new_den_embeddings = torch.matmul(torch.mul(denoise_graph, self.dense_norm_adj[user_idx]), den_all_embeddings[-1])
            den_ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, den_all_embeddings[-1])
            den_ego_embeddings[user_idx] = new_den_embeddings
            den_all_embeddings += [den_ego_embeddings]
        den_all_embeddings = torch.stack(den_all_embeddings, dim=1)
        den_all_embeddings = torch.mean(den_all_embeddings, dim=1)
        den_user_all_embeddings = den_all_embeddings[:self.data.user_num]
        den_item_all_embeddings = den_all_embeddings[self.data.user_num:]
        
        return user_all_embeddings, item_all_embeddings, gen_user_all_embeddings, gen_item_all_embeddings, den_user_all_embeddings, den_item_all_embeddings, mean, std, all_denoise_graph
    
    def cal_cl_loss(self, view1, view2, temp):
        # user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        #return user_cl_loss + item_cl_loss
        return InfoNCE(view1,view2,temp)


