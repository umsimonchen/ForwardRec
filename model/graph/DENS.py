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
# paper: DENS: Disentangled Negative Sampling for Collaborative Filtering. WSDM'23
# https://github.com/Riwei-HEU/DENS

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

class DENS(GraphRecommender):
	def __init__(self, conf, training_set, test_set):
		super(DENS, self).__init__(conf, training_set, test_set)
		args = OptionConf(self.config['DENS'])
		self.n_layers = int(args['-n_layer'])
		self.warmup = int(args['-warmup'])
		self.gamma = float(args['-gamma'])
		self.candidate = int(args['-candidate'])
		self.model = DENS_Encoder(self.data, self.emb_size, self.n_layers)

	def train(self):
		model = self.model.cuda()
		optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
		early_stopping = False
		epoch = 0
		while not early_stopping:
			for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size, n_negs=self.candidate)):
				user_idx, pos_idx, neg_idx = batch
				rec_user_emb, rec_item_emb = model()
				user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
				
				# multi_negative sampling
				shaped_neg_item_emb = neg_item_emb.view(len(user_idx), self.candidate, (self.n_layers+1), self.emb_size) #[batch_size, n_negs, layer+1, dim]
				all_neg_indices = []
				for k in range(self.n_layers+1):
					u = user_emb[:, k, :]
					pos_i = pos_item_emb[:, k, :]
					neg_i = shaped_neg_item_emb[:, :, k, :]
					
					gate_p = torch.matmul(pos_i, model.weight_dict['wp']) + torch.matmul(u, model.weight_dict['wu']) + model.weight_dict['bp']
					gate_p_e = torch.mul(pos_i, torch.sigmoid(gate_p))
				
					gate_n = torch.matmul(neg_i, model.weight_dict['wn']) + torch.matmul(gate_p_e, model.weight_dict['wr']).unsqueeze(1).repeat(1, self.candidate, 1) + model.weight_dict['bn']
					gate_n_e = torch.mul(neg_i, torch.sigmoid(gate_n))
	
					n_e_sel = (1 - min(1, epoch / self.warmup)) * neg_i - gate_n_e
					scores = (torch.mul(u.unsqueeze(1), n_e_sel)).sum(dim=2) #[batch_size, n_negs]
					indices = torch.max(scores, dim=1)[1].detach()
					all_neg_indices.append(indices)

				all_neg_indices = torch.stack(all_neg_indices, dim=1)
				indices = torch.mode(all_neg_indices, dim=1)[0].view(len(user_idx), 1, 1).expand(-1, -1, self.emb_size)
				
				neg_item_emb = torch.gather(shaped_neg_item_emb.mean(dim=2), 1, indices).squeeze(1)
				pos_item_emb = pos_item_emb.mean(dim=1)
				user_emb = user_emb.mean(dim=1)
				
				# cl_loss
				gate_pos = torch.sigmoid(torch.matmul(pos_item_emb, model.weight_dict['wp']) + torch.matmul(user_emb, model.weight_dict['wu']) + model.weight_dict['bp'])
				gated_pos_e_r = torch.mul(pos_item_emb, gate_pos)
				gated_pos_e_ir = pos_item_emb - gated_pos_e_r
				
				gate_neg = torch.sigmoid(torch.matmul(neg_item_emb, model.weight_dict['wn']) + torch.matmul(gated_pos_e_r, model.weight_dict['wr']) + model.weight_dict['bp'])
				gated_neg_e_r = torch.mul(neg_item_emb, gate_neg)
				gated_neg_e_ir = neg_item_emb - gated_neg_e_r
				
				gated_pos_scores_r = torch.sum(torch.mul(user_emb, gated_pos_e_r), axis=1)
				gated_neg_scores_r = torch.sum(torch.mul(user_emb, gated_neg_e_r), axis=-1)  
				
				gated_pos_scores_ir = torch.sum(torch.mul(user_emb, gated_pos_e_ir), axis=1)
				gated_neg_scores_ir = torch.sum(torch.mul(user_emb, gated_neg_e_ir), axis=-1) 

				cl_loss = self.gamma * (torch.mean(torch.log(1 + torch.exp(gated_pos_scores_ir - gated_pos_scores_r))) \
										+ torch.mean(torch.log(1 + torch.exp(gated_neg_scores_r - gated_neg_scores_ir))) \
										+ torch.mean(torch.log(1 + torch.exp(gated_neg_scores_r - gated_pos_scores_r))) \
										+ torch.mean(torch.log(1 + torch.exp(gated_pos_scores_ir - gated_neg_scores_ir)))) / 4
				rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)/self.batch_size
				batch_loss = rec_loss + cl_loss
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

class DENS_Encoder(nn.Module):
	def __init__(self, data, emb_size, n_layers):
		super(DENS_Encoder, self).__init__()
		self.data = data
		self.latent_size = emb_size
		self.layers = n_layers
		self.norm_adj = data.norm_adj
		self.embedding_dict, self.weight_dict = self._init_model()
		self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

	def _init_model(self):
		initializer = nn.init.xavier_uniform_
		embedding_dict = nn.ParameterDict({
			'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
			'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
			})
		weight_dict = nn.ParameterDict({
			'wp': nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size))),
			'wu': nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size))),
			'wn': nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size))),
			'wr': nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size))),
			'bp': nn.Parameter(initializer(torch.empty(1, self.latent_size))),
			'bn': nn.Parameter(initializer(torch.empty(1, self.latent_size))),
			})
		return embedding_dict, weight_dict

	def forward(self):
		ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
		all_embeddings = [ego_embeddings]
		for k in range(self.layers):
			ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
			all_embeddings += [ego_embeddings]
		all_embeddings = torch.stack(all_embeddings, dim=1)
		user_all_embeddings = all_embeddings[:self.data.user_num, :, :]
		item_all_embeddings = all_embeddings[self.data.user_num:, :, :]
		
		# no sampling
		# all_embeddings = all_embeddings.mean(dim=1)
		# user_all_embeddings = all_embeddings[:self.data.user_num, :]
		# item_all_embeddings = all_embeddings[self.data.user_num:, :]
		return user_all_embeddings, item_all_embeddings


