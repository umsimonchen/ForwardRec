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
# paper: DINS: Dimension Independent Mixup for Hard Negative Sample in Collaborative Filtering. CIKM'23
# https://github.com/Wu-Xi/DINS

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

class DINS(GraphRecommender):
	def __init__(self, conf, training_set, test_set):
		super(DINS, self).__init__(conf, training_set, test_set)
		args = OptionConf(self.config['DINS'])
		self.n_layers = int(args['-n_layer'])
		self.alpha = float(args['-alpha'])
		self.candidate = int(args['-candidate'])
		self.K = 1 # default constant
		self.model = DINS_Encoder(self.data, self.emb_size, self.n_layers, self.alpha)

	def train(self):
		model = self.model.cuda()
		optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
		early_stopping = False
		epoch = 0
		while not early_stopping:
			for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size, n_negs=self.candidate*self.K)):
				user, pos_item, neg_item = batch
				neg_item = torch.tensor(neg_item).view(len(user), self.candidate*self.K)
				user_gcn_emb, item_gcn_emb = model()
				
				neg_gcn_embs = []
				for k in range(self.K):
					neg_user_embs = model.negative_sampling(user_gcn_emb, item_gcn_emb, user, neg_item[:, k*self.candidate:(k+1)*self.candidate], pos_item)
					neg_gcn_embs.append(neg_user_embs)
				neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)
				
				u_e = user_gcn_emb[user].mean(dim=1)
				pos_e = item_gcn_emb[pos_item].mean(dim=1)
				neg_e = neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3]).mean(dim=1).view(len(user), self.K, -1)
				
				rec_loss = bpr_loss(u_e.unsqueeze(dim=1), pos_e.unsqueeze(dim=1), neg_e) + l2_reg_loss(self.reg, u_e, pos_e, neg_e)/self.batch_size
				batch_loss = rec_loss
				# Backward and optimize
				optimizer.zero_grad()
				batch_loss.backward()
				optimizer.step()
				if n % 100==0 and n>0:
					print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
			with torch.no_grad():
				self.user_emb, self.item_emb = model()
				self.user_emb = self.user_emb.mean(dim=1)
				self.item_emb = self.item_emb.mean(dim=1)
				_, early_stopping = self.fast_evaluation(epoch)
			epoch += 1
		self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
		with open('performance.txt','a') as fp:
			fp.write(str(self.bestPerformance[1])+"\n")
    
	def save(self):
		with torch.no_grad():
			self.best_user_emb, self.best_item_emb = self.model.forward()
			self.best_user_emb = self.best_user_emb.mean(dim=1)
			self.best_item_emb = self.best_item_emb.mean(dim=1)

	def predict(self, u):
		u = self.data.get_user_id(u)
		score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
		return score.cpu().numpy()

class DINS_Encoder(nn.Module):
	def __init__(self, data, emb_size, n_layers, alpha):
		super(DINS_Encoder, self).__init__()
		self.data = data
		self.latent_size = emb_size
		self.layers = n_layers
		self.alpha = alpha
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
    
	def negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
		batch_size = len(user)
		s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]
		s_e = s_e.mean(dim=1).unsqueeze(dim=1) # default mean pooling
        
		"""Hard Boundary Definition"""
		n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
		scores = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
		indices = torch.max(scores, dim=1)[1].detach()  # torch.Size([2048, 3])
		neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
		neg_items_embedding_hardest = neg_items_emb_[[[i] for i in range(batch_size)],range(neg_items_emb_.shape[1]), indices, :]   #   [batch_size, n_hops+1, channel]

		"""Dimension Independent Mixup"""
		neg_scores = torch.exp(s_e *neg_items_embedding_hardest)  # [batch_size, n_hops, channel]
		total_sum = self.alpha * torch.exp ((s_e * p_e))+neg_scores   # [batch_size, n_hops, channel]
		neg_weight = neg_scores/total_sum     # [batch_size, n_hops, channel]
		pos_weight = 1-neg_weight   # [batch_size, n_hops, channel]

		n_e_ =  pos_weight * p_e + neg_weight * neg_items_embedding_hardest  # mixing
        
		return n_e_
    
	def forward(self):
		ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
		all_embeddings = [ego_embeddings]
		for k in range(self.layers):
			ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
			all_embeddings += [ego_embeddings]
		all_embeddings = torch.stack(all_embeddings, dim=1)
		user_all_embeddings = all_embeddings[:self.data.user_num, :]
		item_all_embeddings = all_embeddings[self.data.user_num:, :]
		
		return user_all_embeddings, item_all_embeddings


