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
# paper: DiffNet: A Neural Influence Diffusion Model for Social Recommendation. SIGIR'19
# https://github.com/PeiJieSun/diffnet

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
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

class DiffNet(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(DiffNet, self).__init__(conf, training_set, test_set, **kwargs)
        args = OptionConf(self.config['DiffNet'])
        self.n_layers = int(args['-n_layer'])
        self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
        self.print_model_info()
        self.social_data_binary = self.social_data.get_social_mat()
        self.social_data_binary = self.social_data_binary.multiply(1.0/self.social_data_binary.sum(axis=1).reshape(-1, 1))
        self.model = DiffNet_Encoder(self.data, self.social_data_binary, self.emb_size, self.n_layers)
    
    def print_model_info(self):
            super(DiffNet, self).print_model_info()
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
                self.user_emb, self.item_emb = model.forward()
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

class DiffNet_Encoder(nn.Module):
    def __init__(self, data, social_data, emb_size, n_layers):
        super(DiffNet_Encoder, self).__init__()
        self.data = data
        self.social_data = social_data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_inter = data.norm_inter
        self.embedding_dict = self._init_model()
        self.sparse_norm_inter = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_inter).cuda()
        self.sparse_social_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.social_data).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = {'weight%d'%k: nn.Parameter(initializer(torch.empty(2*self.latent_size, self.latent_size))) for k in range(self.layers)}
        embedding_dict['user_emb'] = nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size)))
        embedding_dict['item_emb'] = nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size)))
        embedding_dict = nn.ParameterDict(embedding_dict)
        return embedding_dict

    def forward(self):
        user_embeddings = self.embedding_dict['user_emb']
        item_embeddings = self.embedding_dict['item_emb'].clone() # null operation
        for k in range(self.layers):
            new_user_embeddings = torch.sparse.mm(self.sparse_social_adj, user_embeddings)
            user_embeddings = torch.matmul(torch.cat([new_user_embeddings, user_embeddings], 1), self.embedding_dict['weight%d'%k])
            # user_embeddings = torch.max(user_embeddings, torch.Tensor([0]).cuda())
        final_user_embeddings = user_embeddings + torch.sparse.mm(self.sparse_norm_inter, item_embeddings)
        return final_user_embeddings, item_embeddings


