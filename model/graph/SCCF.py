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
# paper: Unifying Graph Convolution and Contrastive Learning in Collaborative Filtering. KDD'24
# https://github.com/wu1hong/SCCF/tree/master

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

class SCCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SCCF, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SCCF'])
        self.n_layers = int(args['-n_layer'])
        self.tau = float(args['-tau'])
        self.model = SCCF_Encoder(self.data, self.emb_size, self.n_layers, self.tau)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        while not early_stopping:
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, _ = batch
                rec_user_emb, rec_item_emb = model()
                
                u_idx, u_inv_idx, u_counts = torch.unique(torch.tensor(user_idx).cuda(), return_counts=True, return_inverse=True)
                i_idx, i_inv_idx, i_counts = torch.unique(torch.tensor(pos_idx).cuda(), return_counts=True, return_inverse=True)
                u_counts, i_counts = u_counts.reshape(-1, 1).float(), i_counts.reshape(-1, 1).float()
                
                user_emb, pos_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx]
                user_emb = torch.nn.functional.normalize(user_emb, dim=-1)
                pos_item_emb = torch.nn.functional.normalize(pos_item_emb, dim=-1)
                ip = (user_emb * pos_item_emb).sum(dim=1)
                up_score = (ip / self.tau).exp() + (ip ** 2 / self.tau).exp()
                up = up_score.log().mean()

                ego_user_emb, ego_pos_item_emb = rec_user_emb[u_idx], rec_item_emb[i_idx]
                ego_user_emb = torch.nn.functional.normalize(ego_user_emb, dim=-1)
                ego_pos_item_emb = torch.nn.functional.normalize(ego_pos_item_emb, dim=-1)
                sim_mat = ego_user_emb @ ego_pos_item_emb.T
                score = (sim_mat / self.tau).exp() + (sim_mat ** 2 / self.tau).exp()
                down = (score * (u_counts @ i_counts.T)).mean().log()
                
                batch_loss = -up + down + l2_reg_loss(self.reg, user_emb, pos_item_emb) 

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

class SCCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, tau):
        super(SCCF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.tau = tau
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings
    
    
    
        


