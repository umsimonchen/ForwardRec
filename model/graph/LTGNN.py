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
import networkx as nx
from data.graph import Graph
import scipy

# paper: LTGNN: Linear-Time Graph Neural Networks for Scalable Recommendations WWW'24
# This paper is heavily based of APPNP. But the hyperparameter of APPNP have not been discussed in LTGNN.
# For reference, we reproduce the code here completely based on data reported in the paper LTGNN. 
# For fair comparison, we conducted experiments on the original codes: https://github.com/QwQ2000/TheWebConf24-LTGNN-PyTorch/tree/main

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

class LTGNN(GraphRecommender):
	def __init__(self, conf, training_set, test_set):
		super(LTGNN, self).__init__(conf, training_set, test_set)
		args = OptionConf(self.config['LTGNN'])
		self.n_layers = int(args['-n_layer'])
		self.alpha = float(args['-alpha'])
		self.num_sample = int(args['-num_sample'])
		self.model = LTGNN_Encoder(self.data, self.emb_size, self.n_layers, self.alpha, self.num_sample)

	def train(self):
		model = self.model.cuda()
		optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
		early_stopping = False
		epoch = 0
		while not early_stopping:
			model.update_memory()
			for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
				user_idx, pos_idx, neg_idx = batch
				rec_user_emb, rec_item_emb = model()
				user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
				batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
				# Backward and optimize
				batch_loss.backward()
				optimizer.step()
				if n % 100==0 and n>0:
					print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
			with torch.no_grad():
				self.user_emb, self.item_emb = model(training=False)
				_, early_stopping = self.fast_evaluation(epoch)
			epoch += 1
		self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
		with open('performance.txt','a') as fp:
			fp.write(str(self.bestPerformance[1])+"\n")
    
	def save(self):
		with torch.no_grad():
			self.best_user_emb, self.best_item_emb = self.model.forward(training=False)
	def predict(self, u):
		u = self.data.get_user_id(u)
		score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
		return score.cpu().numpy()

class LTGNN_Encoder(nn.Module):
	def __init__(self, data, emb_size, n_layers, alpha, num_sample):
		super(LTGNN_Encoder, self).__init__()
		self.data = data
		self.latent_size = emb_size
		self.layers = n_layers
		self.alpha = alpha
		self.num_sample = num_sample
		self.embedding_dict = self._init_model()
		# graph
		self.norm_adj = data.norm_adj
		self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
		self.ui_adj = data.ui_adj
		edges = self.ui_adj.nonzero()
		self.neighbor_dict = {}
		self.neighbor_num = {}
		for i in range(len(edges[0])):
			try:
				self.neighbor_dict[edges[0][i]].append(edges[1][i])
			except:
				self.neighbor_dict[edges[0][i]] = [edges[1][i]]
		for key in self.neighbor_dict:
			self.neighbor_num[key] = len(self.neighbor_dict[key])
		# variable initialize
		self.e_out = None
		self.e_in_g = None
		
	def _init_model(self):
		initializer = nn.init.xavier_uniform_
		embedding_dict = nn.ParameterDict({
			'e_in': nn.Parameter(initializer(torch.empty(self.data.user_num+self.data.item_num, self.latent_size))),
		})
		return embedding_dict

	def generate_subgraph(self, user_idx, pos_idx, neg_idx):
		i_idx = np.unique(pos_idx+neg_idx) + self.data.user_num
		u_idx = np.unique(user_idx)
		all_idx = np.concatenate((i_idx,u_idx), axis=0)
		row = []
		col = []
		val = []
		for idx in all_idx:
			try:
				neigh_sampled = random.sample(self.neighbor_dict[idx], self.num_sample)
				row += [idx] * self.num_sample
				col += neigh_sampled
				val += [1/self.num_sample] * self.num_sample # using num_neighbors would cause gradient explosion
			except:
				neigh_sampled = self.neighbor_dict[idx]
				row += [idx] * len(neigh_sampled)
				col += neigh_sampled
				val += [1/len(neigh_sampled)] * len(neigh_sampled) # using num_neighbors would cause gradient explosion
		row = np.array(row)
		col = np.array(col)
		val = np.array(val)
		#tmp_adj = scipy.sparse.csr_matrix((val, (row, col)), shape=(self.data.user_num+self.data.item_num, self.data.user_num+self.data.item_num),dtype=np.float32)
		i = torch.LongTensor(np.array([row, col]))
		v = torch.FloatTensor(val)
		N = self.data.user_num + self.data.item_num
		return torch.sparse_coo_tensor(i, v, [N, N]).cuda()
		
	def update_memory(self):
		if torch.is_tensor(self.e_out): 
			pass
		else: # first training epoch
			self.e_out = torch.zeros(self.data.user_num+self.data.item_num, self.latent_size).cuda()
		self.in_mem = self.e_out.clone().detach() #[m+n, d]
		self.in_mem.requires_grad_(False) # no_backprop
		
		if torch.is_tensor(self.e_in_g):
			pass
		else:
			self.e_in_g = torch.zeros(self.data.user_num+self.data.item_num, self.latent_size).cuda()
		self.in_mem_g = self.e_in_g.clone().detach()
		self.in_mem_g.requires_grad_(False) # no_backprop

		self.in_aggr_mem = torch.sparse.mm(self.sparse_norm_adj, self.in_mem).clone().detach() #[m+n, d]
		self.in_aggr_mem.requires_grad_(False) # no_backprop
		self.in_aggr_mem_g = torch.sparse.mm(self.sparse_norm_adj, self.in_mem_g).clone().detach()
		self.in_aggr_mem_g.requires_grad_(False) # no_backprop

	def forward(self, user_idx=None, pos_idx=None, neg_idx=None, training=True):
		adj = self.sparse_norm_adj
		if user_idx and pos_idx and neg_idx:
			adj = self.generate_subgraph(user_idx, pos_idx, neg_idx)
		if training==False:
			return self.e_out[:self.data.user_num], self.e_out[self.data.user_num:]
		def backward_hook(grad):
			self.e_in_g = grad
		self.e_in = self.embedding_dict['e_in']
		self.e_in.register_hook(backward_hook)
		self.e_out_new = LTGNN_Layer.apply(adj, self.e_in, self.e_out.detach(), self.alpha, self.in_mem, self.in_aggr_mem, self.in_mem_g, self.in_aggr_mem_g, self.e_in_g)
		self.e_out = self.e_out_new.clone().detach()	
		return self.e_out_new[:self.data.user_num], self.e_out[self.data.user_num:]

class LTGNN_Layer(torch.autograd.Function):
	@staticmethod
	def forward(ctx, adj, e_in, e_out, alpha, in_mem, in_aggr_mem, in_mem_g, in_aggr_mem_g, e_in_g):
		x_evr = torch.sparse.mm(adj, e_out - in_mem) + in_aggr_mem
		e_out_new = (1 - alpha) * x_evr + alpha * e_in
			
		ctx.save_for_backward(adj, torch.tensor(alpha), in_mem_g, in_aggr_mem_g, e_out_new, e_in_g)
		return e_out_new
		
	@staticmethod
	def backward(ctx, grad_e_out_new):
		adj, alpha, in_mem_g, in_aggr_mem_g, e_out_new, e_in_g = ctx.saved_tensors
		
		g_evr = torch.sparse.mm(adj, e_in_g - in_mem_g) + in_aggr_mem_g 
		grad_e_in = (1 - alpha) * g_evr + alpha * grad_e_out_new
		return None, grad_e_in, None, None, None, None, None, None, None














