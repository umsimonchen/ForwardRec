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
        self.n_layers = int(args['-n_layer'])
        self.cl_rate = float(args['-lambda'])
        aug_type = int(args['-augtype'])
        drop_rate = float(args['-droprate'])
        temp = float(args['-temp'])
        self.model = LightGCL_Encoder(self.data, self.emb_size, self.n_layers, aug_type, drop_rate, temp)

    def train(self):
        record_list = []
        loss_list = []
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        while not early_stopping:
            self.dropped_adj = model.graph_reconstruction()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                all_embeddings, all_low_rank_embeddings = model(self.dropped_adj)
                
                final_all_embeddings = torch.sum(all_embeddings, dim=0)
                rec_user_emb = final_all_embeddings[:self.data.user_num]
                rec_item_emb = final_all_embeddings[self.data.user_num:]
                
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                cl_loss = 0
                for k in range(self.n_layers):
                    cl_loss += self.cl_rate * model.cal_cl_loss([user_idx], all_embeddings[k][:self.data.user_num], all_low_rank_embeddings[k][:self.data.user_num])
                    cl_loss += self.cl_rate * model.cal_cl_loss([pos_idx], all_embeddings[k][self.data.user_num:], all_low_rank_embeddings[k][self.data.user_num:])
                total_loss = batch_loss + cl_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                all_embeddings, _ = model(self.dropped_adj)
                all_embeddings = torch.sum(all_embeddings, dim=0)
                self.user_emb = all_embeddings[:self.data.user_num]
                self.item_emb = all_embeddings[self.data.user_num:]
                
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
            all_embeddings, _ = self.model.forward(self.dropped_adj)
            all_embeddings = torch.sum(all_embeddings, dim=0)
            self.best_user_emb = all_embeddings[:self.data.user_num]
            self.best_item_emb = all_embeddings[self.data.user_num:]

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class LightGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, aug_type, drop_rate, temp):
        super(LightGCL_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.aug_type = aug_type
        self.drop_rate = drop_rate
        self.temp = temp
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict
    
    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()
    
    def forward(self, dropped_adj):
        
        # please get the eigenvalues and eigenvectors by eigendecomp.py
        with open('dataset/yelp2018/PGSP', 'rb') as fp: # choose the dataset folder
            eigen = pickle.load(fp)
        val = eigen[0]
        vec = eigen[1]
        self.e = torch.tensor(vec).cuda().float()
        self.v = torch.tensor(val).cuda().float()
        
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        all_low_rank_embeddings = [ego_embeddings]
        for k in range(self.layers):
            # ego_embeddings = nn.LeakyReLU(0.5)(torch.sparse.mm(norm_dropped_adj, all_embeddings[-1]))
            # all_embeddings += [ego_embeddings + all_embeddings[-1]]
            
            # ego_low_rank_embeddings = nn.LeakyReLU(0.5)(torch.matmul(low_rank_graph, all_low_rank_embeddings[-1]))
            # all_low_rank_embeddings = [ego_low_rank_embeddings]
            
            # LightGCN based message passing performs better after comparison
            ego_embeddings = torch.sparse.mm(dropped_adj, all_embeddings[-1])
            all_embeddings += [ego_embeddings]
            
            ego_low_rank_embeddings = torch.matmul(torch.matmul(self.e, torch.diag(self.v)), torch.matmul(self.e.transpose(0, 1), all_low_rank_embeddings[-1]))
            all_low_rank_embeddings += [ego_low_rank_embeddings]
            
        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_low_rank_embeddings = torch.stack(all_low_rank_embeddings, dim=0)
        
        return all_embeddings, all_low_rank_embeddings
    
    def cal_cl_loss(self, idx, view1, view2):
        idx = torch.unique(torch.Tensor(idx).type(torch.long)).cuda()
        # user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        #return user_cl_loss + item_cl_loss
        return InfoNCE(view1[idx],view2[idx],self.temp)


