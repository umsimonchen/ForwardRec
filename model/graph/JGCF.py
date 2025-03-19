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
# https://github.com/SpaceLearner/JGCF/tree/main

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
        self.layers = int(args['-layers'])
        self.a = float(args['-a'].replace('"',''))
        self.b = float(args['-b'])
        self.beta = float(args['-alpha'])
        self.model = JGCF_Encoder(self.data, self.layers, self.emb_size, self.a, self.b, self.beta)

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
                u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings = \
                    model.embedding_dict['user_emb'][user_idx], model.embedding_dict['item_emb'][pos_idx], model.embedding_dict['item_emb'][neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)/self.batch_size
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
        # with open('training_record','wb') as fp:
        #     pickle.dump([record_list, loss_list], fp)

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class JGCF_Encoder(nn.Module):
    def __init__(self, data, layers, emb_size, a, b, beta):
        super(JGCF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = layers
        self.a = a
        self.b = b
        self.beta = beta
        self.basealpha = 3.0 # default constant for 3 layers
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(float(min(1 / self.basealpha, 1))),
                         requires_grad=not True) for i in range(self.layers + 1)
        ])
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

    def JacobiConv(self, L, xs, adj, alphas, a=1.0, b=1.0, l=-1.0, r=1.0):
        if L == 0: return xs[0]
        if L == 1:
            coef1 = (a - b) / 2 - (a + b + 2) / 2 * (l + r) / (r - l)
            coef1 *= alphas[0]
            coef2 = (a + b + 2) / (r - l)
            coef2 *= alphas[0]
            return coef1 * xs[-1] + coef2 * (adj @ xs[-1])
        coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)
        coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
        coef_lm1_2 = (2 * L + a + b - 1) * (a**2 - b**2)
        coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)
        tmp1 = alphas[L - 1] * (coef_lm1_1 / coef_l)
        tmp2 = alphas[L - 1] * (coef_lm1_2 / coef_l)
        tmp3 = alphas[L - 1] * alphas[L - 2] * (coef_lm2 / coef_l)
        tmp1_2 = tmp1 * (2 / (r - l))
        tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2
        nx = tmp1_2 * (adj @ xs[-1]) - tmp2_2 * xs[-1]
        nx -= tmp3 * xs[-2]
        return nx
    
    def forward(self):
        all_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        alphas = [self.basealpha * torch.tanh(_) for _ in self.alphas]
        xs = [self.JacobiConv(0, [all_embeddings], self.sparse_norm_adj, alphas, self.a, self.b)]
        for L in range(1, self.layers + 1):
            tx = self.JacobiConv(L, xs, self.sparse_norm_adj, alphas, self.a, self.b)
            xs.append(tx)
        xs = [x.unsqueeze(1) for x in xs]
        all_embeddings_low = torch.cat(xs, dim=1)
        all_embeddings_low = all_embeddings_low.mean(1)
        all_embeddings_mid = self.beta * all_embeddings - all_embeddings_low
        all_embeddings = torch.hstack([all_embeddings_low, all_embeddings_mid])

        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        
        return user_all_embeddings, item_all_embeddings


