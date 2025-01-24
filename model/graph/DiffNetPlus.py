import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss
from data.social import Relation
import os
import numpy as np
import random
# paper: DiffNet++: A Neural Influence and Interest Diffusion Network for Social Recommendation. TKDE'22
# https://github.com/PeiJieSun/diffnet/tree/master/Diffnet++

seed = 0
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

class DiffNetPlus(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(DiffNetPlus, self).__init__(conf, training_set, test_set, **kwargs)
        args = OptionConf(self.config['DiffNetPlus'])
        self.n_layers = int(args['-n_layer'])
        self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
        self.print_model_info()
        self.social_data_binary = self.social_data.get_social_mat()
        self.social_data_binary = self.social_data_binary.multiply(1.0/self.social_data_binary.sum(axis=1).reshape(-1, 1))
        self.model = DiffNetPlus_Encoder(self.data, self.social_data_binary, self.emb_size, self.n_layers)
    
    def print_model_info(self):
        super(DiffNetPlus, self).print_model_info()
        # # print social relation statistics
        print('Social data size: (user number: %d, relation number: %d).' % (self.social_data.size()))
        print('=' * 80)
        
    def train(self):
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
            _, early_stopping = self.fast_evaluation(epoch)
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

class DiffNetPlus_Encoder(nn.Module):
    def __init__(self, data, social_data, emb_size, n_layers):
        super(DiffNetPlus_Encoder, self).__init__()
        self.data = data
        self.social_data = social_data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_inter = data.norm_inter
        self.embedding_dict = self._init_model()
        self.sparse_norm_inter = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_inter).cuda()
        self.sparse_social_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.social_data.tocsr()).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = {'weight%d'%k: nn.Parameter(initializer(torch.empty(3*self.latent_size, self.latent_size))) for k in range(self.layers)}
        embedding_dict['user_emb'] = nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size)))
        embedding_dict['item_emb'] = nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size)))
        embedding_dict = nn.ParameterDict(embedding_dict)
        return embedding_dict

    def forward(self):
        all_user_embeddings = [self.embedding_dict['user_emb']]
        all_item_embeddings = [self.embedding_dict['item_emb']]
        final_user_embeddings = self.embedding_dict['user_emb']
        final_item_embeddings = self.embedding_dict['item_emb']
        for k in range(self.layers):
            new_user_social_embeddings = torch.sparse.mm(self.sparse_social_adj, all_user_embeddings[k])
            new_user_inter_embeddings = torch.sparse.mm(self.sparse_norm_inter, all_item_embeddings[k])
            new_user_embeddings = torch.matmul(torch.cat([new_user_social_embeddings, new_user_inter_embeddings, all_user_embeddings[k]], 1), self.embedding_dict['weight%d'%k])
            new_item_embeddings = torch.sparse.mm(self.sparse_norm_inter.transpose(0, 1), all_user_embeddings[k])
            new_user_embeddings = nn.ReLU()(new_user_embeddings)
            new_item_embeddings = nn.ReLU()(new_item_embeddings)
            all_user_embeddings.append(new_user_embeddings)
            all_item_embeddings.append(new_item_embeddings)
            final_user_embeddings = torch.cat([final_user_embeddings, new_user_embeddings], 1)
            final_item_embeddings = torch.cat([final_item_embeddings, new_item_embeddings], 1)
            
        return final_user_embeddings, final_item_embeddings



