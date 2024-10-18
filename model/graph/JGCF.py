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
# paper: JGCF: On Manipulating Signals of User-Item Graph: A Jacobi Polynomial-based Graph Collaborative Filtering, KDD'23

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

class JGCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(JGCF, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['JGCF'])
        self.n_layers = int(args['-n_layer'])
        self.a = float(args['-a'])
        self.b = float(args['-b'])
        self.alpha = float(args['-alpha'])
        self.model = JGCF_Encoder(self.data, self.emb_size, self.n_layers, self.a, self.b, self.alpha)

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
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
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
        # record training loss        
        with open('training_record','wb') as fp:
            pickle.dump([record_list, loss_list], fp)

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class JGCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, a, b, alpha):
        super(JGCF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.a = a
        self.b = b
        self.a_mat = a*torch.ones(self.data.user_num+self.data.item_num, self.data.user_num+self.data.item_num).cuda()
        self.b_mat = b*torch.ones(self.data.user_num+self.data.item_num, self.data.user_num+self.data.item_num).cuda()
        self.alpha = alpha
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.dense_norm_adj = self.sparse_norm_adj.to_dense()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        jacobi_all_embeddings = [ego_embeddings]
        jacobi_all_basis = [torch.eye(self.data.user_num+self.data.item_num).cuda()]
        r = 1.0
        l = -1.0
        for k in range(self.layers):
            cur_layer = k+1
            if cur_layer == 1:
                jacobi_all_basis += [(self.a-self.b)/2 + (self.a+self.b + 2)/2 * self.dense_norm_adj]
                jacobi_all_embeddings += [(self.a-self.b)/2 * jacobi_all_embeddings[0] + (self.a+self.b+2)/2 * torch.sparse.mm(self.sparse_norm_adj, jacobi_all_embeddings[0])]
            else:
                # theta = (2*cur_layer+self.a+self.b) * (2*cur_layer+self.a+self.b-1) / (2*cur_layer*(cur_layer+self.a+self.b))
                # theta_p = (2*cur_layer+self.a+self.b-1) * (self.a**2 - self.b**2) / (2*cur_layer*(cur_layer+self.a+self.b)*(2*cur_layer+self.a+self.b-2))
                # theta_pp = (cur_layer+self.a-1) * (cur_layer+self.b-1) * (2*cur_layer+self.a+self.b) / (cur_layer*(cur_layer+self.a+self.b)*(2*cur_layer+self.a+self.b-2))
                # jacobi_all_basis += [theta*(torch.sparse.mm(self.sparse_norm_adj, jacobi_all_basis[-1])) + theta_p*jacobi_all_basis[-1] - theta_pp*jacobi_all_basis[-2]]
                # jacobi_all_embeddings += [jacobi_all_basis[-1]@jacobi_all_embeddings[0]]
                coef_l = 2 * cur_layer * (cur_layer + self.a + self.b) * (2 * cur_layer - 2 + self.a + self.b)
                coef_lm1_1 = (2 * cur_layer + self.a + self.b - 1) * (2 * cur_layer + self.a + self.b) * (2 * cur_layer + self.a + self.b - 2)
                coef_lm1_2 = (2 * cur_layer + self.a + self.b - 1) * (self.a**2 - self.b**2)
                coef_lm2 = 2 * (cur_layer - 1 + self.a) * (cur_layer - 1 + self.b) * (2 * cur_layer + self.a + self.b)
                theta = coef_lm1_1 / coef_l            
                theta_p = coef_lm1_2 / coef_l
                theta_pp = coef_lm2 / coef_l
                #tmp1_2 = tmp1 * (2 / (r - l))
                #tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2
                jacobi_all_basis += [theta * torch.sparse.mm(self.sparse_norm_adj, jacobi_all_basis[-1]) + theta_p * jacobi_all_basis[-1] - theta_pp * jacobi_all_basis[-2]]
                jacobi_all_embeddings += [jacobi_all_basis[-1]@jacobi_all_embeddings[0]]
      
        band_stop = torch.mean(torch.stack(jacobi_all_embeddings, dim=1), dim=1)
        band_pass = torch.tanh(self.alpha*jacobi_all_embeddings[0] - band_stop)
        final_embeddings = torch.cat([band_stop, band_pass], dim=1)
        user_all_embeddings = final_embeddings[:self.data.user_num]
        item_all_embeddings = final_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


