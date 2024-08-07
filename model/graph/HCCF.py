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
from data.augmentor import GraphAugmentor
# paper: Hypergraph Contrastive Collaborative Filtering. SIGIR'22
# https://github.com/akaxlh/HCCF/tree/main

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

class HCCF(GraphRecommender):
  def __init__(self, conf, training_set, test_set):
    super(HCCF, self).__init__(conf, training_set, test_set)
    args = OptionConf(self.config['HCCF'])
    self.n_layers = int(args['-n_layer'])
    self.hyperedge = int(args['-hyperedge'])
    self.droprate = float(args['-droprate'])
    self.cl_rate = float(args['-cl_rate'])
    self.model = HCCF_Encoder(self.data, self.emb_size, self.n_layers, self.hyperedge, self.droprate)

  def train(self):
    model = self.model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
    early_stopping = False
    epoch = 0
    while not early_stopping:
      for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
        user_idx, pos_idx, neg_idx = batch
        rec_user_emb, rec_item_emb, all_z, all_gamma = model()
        user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
        #cl loss
        user_cl = 0
        item_cl = 0
        for k in range(self.n_layers):
          u_idx = torch.unique(torch.Tensor(user_idx).type(torch.long)).cuda()
          i_idx = torch.unique(torch.Tensor(pos_idx).type(torch.long)).cuda()
          z_user, z_item = torch.split(all_z[k], [self.data.user_num, self.data.item_num], dim=0)
          gamma_user, gamma_item = torch.split(all_gamma[k], [self.data.user_num, self.data.item_num], dim=0)
          
          z_user = nn.functional.normalize(z_user, p=2, dim=1)
          gamma_user = nn.functional.normalize(gamma_user, p=2, dim=1)
          nume = torch.exp(torch.mul(z_user, gamma_user).sum(dim=1, keepdim=True) / 0.2)
          deno = torch.exp(torch.matmul(z_user, gamma_user.transpose(0, 1)) / 0.2).sum(dim=1, keepdim=True)
          user_cl += -torch.log(nume/deno).mean()
          
          z_item = nn.functional.normalize(z_item[i_idx], p=2, dim=1)
          gamma_item = nn.functional.normalize(gamma_item[i_idx], p=2, dim=1)
          nume = torch.exp(torch.mul(z_item, gamma_item).sum(dim=1, keepdim=True) / 0.2)
          deno = torch.exp(torch.matmul(z_item, gamma_item.transpose(0, 1)) / 0.2).sum(dim=1, keepdim=True)
          item_cl += -torch.log(nume/deno).mean()

        rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        reg_loss = l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
        cl_loss = self.cl_rate*(user_cl+item_cl)
        batch_loss = rec_loss + reg_loss + cl_loss
        # Backward and optimize
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if n % 100==0 and n>0:
          print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
      with torch.no_grad():
        self.user_emb, self.item_emb, _, _ = model()
      _, early_stopping = self.fast_evaluation(epoch)
      epoch += 1
    self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
    with open('performance.txt','a') as fp:
      fp.write(str(self.bestPerformance[1])+"\n")

  def save(self):
    with torch.no_grad():
      self.best_user_emb, self.best_item_emb, _, _ = self.model.forward()

  def predict(self, u):
    u = self.data.get_user_id(u)
    score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
    return score.cpu().numpy()


class HCCF_Encoder(nn.Module):
  def __init__(self, data, emb_size, n_layers, hyperedge, droprate):
    super(HCCF_Encoder, self).__init__()
    self.data = data
    self.latent_size = emb_size
    self.layers = n_layers
    self.hyperedge = hyperedge
    self.droprate = droprate
    self.norm_adj = data.norm_adj
    self.embedding_dict = self._init_model()
    # masked adjacency matrix
    # since fixed random seed, every time the masked adj is the same, we preprocess it once
    self.masked_norm_adj = GraphAugmentor.edge_dropout(self.norm_adj, self.droprate)
    self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.masked_norm_adj).cuda()

  def _init_model(self):
    initializer = nn.init.xavier_uniform_
    embedding_dict = nn.ParameterDict({
    'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
    'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
    'user_hyper_emb': nn.Parameter(initializer(torch.empty(self.latent_size, self.hyperedge))),
    'item_hyper_emb': nn.Parameter(initializer(torch.empty(self.latent_size, self.hyperedge))),
    })
    return embedding_dict

  def forward(self):
    ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], dim=0)
    all_ego_embeddings = [ego_embeddings] #(m+n)*d
    all_z = []
    all_gamma = []
    hyper_user = torch.matmul(self.embedding_dict['user_emb'], self.embedding_dict['user_hyper_emb']) #m*H
    hyper_item = torch.matmul(self.embedding_dict['item_emb'], self.embedding_dict['item_hyper_emb']) #n*H
    for k in range(self.layers):
      # local
      z = torch.sparse.mm(self.sparse_norm_adj, all_ego_embeddings[k])#(m+n)*d
      all_z += [z]

      # global
      hyper_user_aug = nn.Dropout(p=self.droprate)(hyper_user) #m*H
      hyper_item_aug = nn.Dropout(p=self.droprate)(hyper_item) #n*H
      _lambda_user = torch.matmul(hyper_user_aug.transpose(0, 1), all_ego_embeddings[k][:self.data.user_num]) #H*d
      _lambda_item = torch.matmul(hyper_item_aug.transpose(0, 1), all_ego_embeddings[k][self.data.user_num:]) #H*d
      
      gamma_user = torch.matmul(hyper_user_aug, _lambda_user) #m*d
      gamma_item = torch.matmul(hyper_item_aug, _lambda_item) #n*d
      gamma = torch.cat([gamma_user, gamma_item], dim=0)
      all_gamma += [gamma]
      all_ego_embeddings += [(z+gamma)/2]

    final_embeddings = torch.mean(torch.stack(all_ego_embeddings, dim=1), dim=1)
    final_user_embeddings, final_item_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num], dim=0)

    return final_user_embeddings, final_item_embeddings, all_z, all_gamma


















