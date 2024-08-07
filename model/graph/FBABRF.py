import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss

import networkx as nx
import numpy as np
import time
import math
import os
import random

''' reproducibility settings'''
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

'''debug mode'''
debug_mode = False
if debug_mode:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = 'cpu'
else:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    device = 'cuda'

class FBABRF(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(FBABRF, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['FBABRF'])
        self.n_layers = int(args['-n_layer'])
        self.max_len = int(args['-max_len'])
        self.sample_ratio = float(args['-sample_ratio'])
        self.model = FBABRF_Encoder(self.data, self.emb_size, self.n_layers, self.max_len, self.sample_ratio)

    def train(self):
        model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        G = nx.from_scipy_sparse_array(self.data.norm_adj)
        import pickle
        pickle.dump(G,open('graph','wb'))
        self.paths_dict, self.lengths_dict, self.degrees_dict = self.path_generator(G)
        
        '''batch-training'''
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                paths, lengths, degrees = self.path_sampler(user_idx, pos_idx+neg_idx)
                user_emb, pos_item_emb, neg_item_emb = model(paths, lengths, degrees, True)
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                '''Backward and optimize'''
                optimizer.zero_grad()
                batch_loss.backward(retain_graph=True)
                optimizer.step()
                print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item()*self.batch_size)
            with torch.no_grad():
                paths, lengths, degrees = self.path_sampler()
                self.user_emb, self.item_emb = model(paths, lengths, degrees, False)
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def path_generator(self, G):
        s = time.time()
        sampled_path_dict = {}
        sampled_path_length_dict = {}
        sampled_node_degree_dict = G.degree()
        padding = self.data.user_num + self.data.item_num
        
        # ''' [item->user] '''
        # ''' user path '''
        # for i in range(self.data.user_num):
        #     all_neighbors = list(G.neighbors(i))
        #     sampled_path = []
        #     sampled_path_length = []
        #     for source_node in all_neighbors:
        #         path = nx.shortest_path(G, source_node, i)
        #         sampled_path_length.append(len(path))
        #         path += [padding] * (self.max_len-len(path))
        #         sampled_path.append(path)
        #     sampled_path_dict[i] = sampled_path
        #     sampled_path_length_dict[i] = sampled_path_length
        
        # ''' item path '''
        # for i in range(self.data.item_num):
        #     all_neighbors = list(G.neighbors(i+self.data.user_num))
        #     sampled_path = []
        #     sampled_path_length = []
        #     for source_node in all_neighbors:
        #         path = nx.shortest_path(G, source_node, i+self.data.user_num)
        #         sampled_path_length.append(len(path))
        #         path += [padding] * (self.max_len-len(path))
        #         sampled_path.append(path)
        #     sampled_path_dict[i+self.data.user_num] = sampled_path
        #     sampled_path_length_dict[i+self.data.user_num] = sampled_path_length
        
        # ''' [item->user->item->...->user] '''
        # ''' user path '''
        # for i in range(self.data.user_num):
        #     all_neighbors = list(G.neighbors(i))
        #     sampled_path = []
        #     sampled_path_length = []
        #     for neighbor in all_neighbors:
        #         sampled_path.append(neighbor)
        #         sampled_path.append(i)
        
        ''' random walk '''
        ''' user path '''
                    
        
        e = time.time()
        print("Running time: %f s" % (e - s))
        
        return sampled_path_dict, sampled_path_length_dict, sampled_node_degree_dict #[B,LEN],[B],[#(user_idx)+#(item_idx)]
    
    def path_sampler(self, user_idx=None, item_idx=None):
        paths = []
        lengths = []
        degrees = []
        
        if user_idx and item_idx: # training phase
            for i in user_idx:
                paths += self.paths_dict[i]
                lengths += self.lengths_dict[i]
                degrees += [self.degrees_dict[i]]
            for i in item_idx:
                paths += self.paths_dict[i+self.data.user_num]
                lengths += self.lengths_dict[i+self.data.user_num]
                degrees += [self.degrees_dict[i+self.data.user_num]]
        
        else: # testing phase
            for i in range(self.data.user_num):
                paths += self.paths_dict[i]
                lengths += self.lengths_dict[i]
                degrees += [self.degrees_dict[i]]
            for i in range(self.data.item_num):
                paths += self.paths_dict[i+self.data.user_num]
                lengths += self.lengths_dict[i+self.data.user_num]
                degrees += [self.degrees_dict[i+self.data.user_num]]
        
        return paths, lengths, degrees
    
    def save(self):
        with torch.no_grad():
            paths, lengths, degrees = self.path_sampler()
            self.best_user_emb, self.best_item_emb = self.model.forward(paths, lengths, degrees, False)

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class FBABRF_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, max_len, sample_ratio):
        super(FBABRF_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.max_len = max_len 
        self.heads = 2 # "need to be inpput argument later" supposed to be divisible
        self.each_size = self.latent_size // self.heads
        self.embedding_dict = self._init_model()
        self.frequency_len = self.max_len // 2 + 1
        self.sample_ratio = sample_ratio
        self.std = True # "need to be input argument later"
        self.dual_domain = True # "need to be input argument later"
        self.factor = 1 # "need to be input argument later"
        self.spatial_ratio = 0.1 # "need to be input argument later"
        if self.sample_ratio > (1/self.layers):
            self.step = (self.frequency_len * (1-self.sample_ratio)) // (self.layers-1)
        else:
            self.sample_ratio = 1/self.layers
            self.step = (self.frequency_len / self.layers)
    
    def get_bi_attention_mask(self, paths):
        attention_mask = (paths < (self.data.user_num+self.data.item_num)).long() # [B,LEN]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # [B,1,1,LEN]
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask 
    
    def HybridAttention(self, k, input_embeddings, attention_mask):
        # initialization
        left = int((self.frequency_len * (1-self.sample_ratio)) - (k*self.step)) # left is high frequency
        right = int(self.frequency_len - k*self.step) # right is left frequency
        q_index = list(range(left, right))
        k_index = list(range(left, right))
        v_index = list(range(left, right))
        
        time_q_index = q_index if self.std else list(range(self.frequency_len)) # standard mode to use same length as q_index
        time_k_index = k_index if self.std else list(range(self.frequency_len)) # else use whole frequency length
        time_v_index = v_index if self.std else list(range(self.frequency_len))
        
        '''go through linear layers respectively'''
        mixed_query_layer = torch.matmul(input_embeddings, self.embedding_dict['query_layer%d'%k][0].to(device))
        mixed_key_layer = torch.matmul(input_embeddings, self.embedding_dict['key_layer%d'%k][0].to(device))
        mixed_value_layer = torch.matmul(input_embeddings, self.embedding_dict['value_layer%d'%k][0].to(device))
        
        '''divide by heads: [B,LEN,H,E]'''
        queries = mixed_query_layer.view(self.batch_num, self.max_len, self.heads, self.each_size) 
        keys = mixed_key_layer.view(self.batch_num, self.max_len, self.heads, self.each_size)
        values = mixed_value_layer.view(self.batch_num, self.max_len, self.heads, self.each_size)
        
        '''period-based dependencies, rfft will divide the signal length: [B,H,E,M]'''
        '''M = self.max_len // 2 + 1 # only need half of the sequence'''
        q_fft = torch.fft.rfft(queries.permute(0,2,3,1).contiguous(), dim=-1) # M = LEN//2 + 1
        k_fft = torch.fft.rfft(keys.permute(0,2,3,1).contiguous(), dim=-1)
        v_fft = torch.fft.rfft(values.permute(0,2,3,1).contiguous(), dim=-1)
        
        '''sample in frequency domain'''
        q_fft_box = torch.zeros(self.batch_num, self.heads, self.each_size, len(q_index), device=q_fft.device, dtype=torch.cfloat)
        k_fft_box = torch.zeros(self.batch_num, self.heads, self.each_size, len(k_index), device=k_fft.device, dtype=torch.cfloat)
        for t,s in enumerate(q_index): # retrieve
            q_fft_box[:,:,:,t] = q_fft[:,:,:,s]
            k_fft_box[:,:,:,t] = k_fft[:,:,:,s]
        
        '''frequency domain attention'''
        res = q_fft_box * torch.conj(k_fft_box) # conjugate, [B,H,E,Step]
        box_res = torch.zeros(self.batch_num, self.heads, self.each_size, self.frequency_len, device=q_fft.device, dtype=torch.cfloat)
        for s,t in enumerate(q_index): # return
            box_res[:,:,:,t] = res[:,:,:,s]
        corr = torch.fft.irfft(box_res, n=self.max_len, dim=-1) # [B,H,E,LEN]
        
        '''time delay agg'''
        topk = max(int(self.factor * math.log(self.max_len)), 1) # in case of 0
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1) # [B,H,E,LEN]->[B,E,LEN]->[B,LEN]
        frequency_layer = torch.zeros(self.batch_num, self.heads, self.each_size, self.max_len).to(device) # [B,H,E,LEN]
        
        if self.training:
            delays = torch.topk(torch.mean(mean_value, dim=0), topk, dim=-1)[1]# [k], speed up for batch-wise topk-index searching
            weights = torch.stack([mean_value[:, delays[tau_i]] for tau_i in range(topk)], dim=-1) # [B,k]
            tmp_corr = torch.softmax(weights, dim=-1) # [B,k]
            for tau_i in range(topk):
                pattern = torch.roll(values.permute(0, 2, 3, 1).contiguous(), -int(delays[tau_i]), -1).to(frequency_layer.device) # values->[B,H,E,LEN]
                frequency_layer = frequency_layer + pattern*(tmp_corr[:,tau_i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, self.heads, self.each_size, self.max_len))
        else:
            weights, delays = torch.topk(mean_value, topk, dim=-1) # [B,k]
            init_index = torch.arange(self.max_len).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(self.batch_num, self.heads, self.each_size, 1).to(values.device)
            tmp_corr = torch.softmax(weights, dim=-1)
            tmp_values = values.permute(0, 2, 3, 1).contiguous().repeat(1, 1, 1, 2) # values->[B,H,E,LEN]
            for tau_i in range(topk):
                tmp_delay = init_index + delays[:, tau_i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, self.heads, self.each_size, self.max_len)
                pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
                frequency_layer = frequency_layer + pattern*(tmp_corr[:,tau_i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, self.heads, self.each_size, self.max_len))
        frequency_layer = frequency_layer.permute(0, 3, 1, 2).view(self.batch_num, self.max_len, self.latent_size) # [B,LEN,H,E]->[B,LEN,D]
        
        '''time domain attention'''
        if self.dual_domain:
            spatial_q = torch.zeros(self.batch_num, self.heads, self.each_size, self.frequency_len, device=q_fft.device, dtype=torch.cfloat) # [B,H,E,M]
            spatial_k = torch.zeros(self.batch_num, self.heads, self.each_size, self.frequency_len, device=k_fft.device, dtype=torch.cfloat)
            spatial_v = torch.zeros(self.batch_num, self.heads, self.each_size, self.frequency_len, device=v_fft.device, dtype=torch.cfloat)
            
            for _,s in enumerate(time_q_index):
                spatial_q[:,:,:,s] = q_fft[:,:,:,s] # only retrive part(all) of fft
                spatial_k[:,:,:,s] = k_fft[:,:,:,s]
                spatial_v[:,:,:,s] = v_fft[:,:,:,s]
            
            queries = torch.fft.irfft(spatial_q, n=self.max_len, dim=-1).permute(0,1,3,2) # [B,H,LEN,E]
            keys = torch.fft.irfft(spatial_k, n=self.max_len, dim=-1).permute(0,1,3,2)
            values = torch.fft.irfft(spatial_v, n=self.max_len, dim=-1).permute(0,1,3,2)
            
            att_scores = torch.matmul(queries, keys.transpose(-1, -2)) # [B,H,LEN,LEN]
            att_scores = att_scores / math.sqrt(self.each_size)
            att_scores = att_scores + attention_mask # [B,H,LEN,LEN] by broadcasting
            att_scores = torch.softmax(att_scores, dim=-1)
            qkv = torch.matmul(att_scores, values).permute(0,2,1,3).contiguous() # [B,H,LEN,E]->[B,LEN,H,E]
            spatial_layer = qkv.view(self.batch_num, self.max_len, self.latent_size) # [B,LEN,D]
            
            final_embeddings = (1-self.spatial_ratio) * frequency_layer + self.spatial_ratio * spatial_layer
        else:
            final_embeddings = frequency_layer
        
        final_embeddings = torch.matmul(final_embeddings, self.embedding_dict['predict_dense%d'%k][0].to(device)) #???why here is the dense matrix and before the gelu
        final_embeddings = final_embeddings + input_embeddings # [B,LEN,D]
        
        return final_embeddings
        
    def _init_model(self):
        init = nn.init.xavier_uniform_ # xavier initializer can improve/accelerate performance
        embedding_dict = {}
        embedding_dict['ego_emb'] = nn.Parameter(init(torch.empty(self.data.user_num+self.data.item_num+1, self.latent_size)))
        embedding_dict['pos_emb'] = nn.Parameter(init(torch.empty(self.max_len, self.latent_size)))
        for k in range(self.layers): #???why all parameters in loop is tupe
            embedding_dict['query_layer%d'%k] = nn.Parameter(init(torch.empty(self.latent_size, self.latent_size))),
            embedding_dict['key_layer%d'%k] = nn.Parameter(init(torch.empty(self.latent_size, self.latent_size))),
            embedding_dict['value_layer%d'%k] = nn.Parameter(init(torch.empty(self.latent_size, self.latent_size))),
            embedding_dict['predict_dense%d'%k] = nn.Parameter(init(torch.empty(self.latent_size, self.latent_size))),
            embedding_dict['ffn_dense1%d'%k] = nn.Parameter(init(torch.empty(self.latent_size, 2*self.latent_size))),
            embedding_dict['ffn_dense2%d'%k] = nn.Parameter(init(torch.empty(2*self.latent_size, self.latent_size))),
        embedding_dict = nn.ParameterDict(embedding_dict) #ego_emb combines [user, item, padding]; pos_emb refers to position embedding
        
        return embedding_dict

    def forward(self, paths, lengths, degrees, training):
        paths = torch.tensor(paths, dtype=torch.int64, device=torch.device(device))
        lengths = torch.tensor(lengths, dtype=torch.int64, device=torch.device(device))
        self.training = training
        
        self.batch_num = paths.shape[0]
        position_embeddings = self.embedding_dict['pos_emb'].unsqueeze(0).repeat([paths.shape[0],1,1]) # [B,LEN]
        path_embeddings = self.embedding_dict['ego_emb'][paths]
        input_embeddings = path_embeddings + position_embeddings
        extended_attention_mask = self.get_bi_attention_mask(paths) # [B,1,1,LEN]
        
        '''FEARec Encoder'''
        all_embeddings = [input_embeddings] # [K,B,LEN,D]
        for k in range(self.layers):
            attention_output = self.HybridAttention(k, all_embeddings[k], extended_attention_mask) # [B,LEN,D]
            
            '''FFN section'''
            attention_output = torch.matmul(attention_output, self.embedding_dict['ffn_dense1%d'%k][0].to(device)) # [B,LEN,2D]
            attention_output = attention_output * 0.5 * (1.0 + torch.erf(attention_output / math.sqrt(2.0)))
            attention_output = torch.matmul(attention_output,self.embedding_dict['ffn_dense2%d'%k][0].to(device)) # [B,LEN,D]
            
            all_embeddings.append(attention_output)
        
        res_embeddings = all_embeddings[-1]
        gather_index = (lengths-1).view(-1, 1, 1).expand(-1, -1, self.latent_size) # [B,1,D], since we padding with last value and minus 1 in case overflow
        res_embeddings = res_embeddings.gather(dim=1, index=gather_index).squeeze(1) # [B,1,D]->[B,D]
        
        '''post-process'''
        splited_embeddings = torch.split(res_embeddings, degrees, dim=0)
        ego_embeddings = []
        for i in range(len(splited_embeddings)):
            ego_embeddings.append(torch.mean(splited_embeddings[i], dim=0))
        ego_embeddings = torch.stack(ego_embeddings)
        if self.training:
            user_embeddings, pos_embeddings, neg_embeddings = torch.split(ego_embeddings, [len(degrees)//3, len(degrees)//3, len(degrees)//3], dim=0)
            return user_embeddings, pos_embeddings, neg_embeddings
        else:
            user_embeddings, item_embeddings = torch.split(ego_embeddings, [self.data.user_num, self.data.item_num], dim=0)
            return user_embeddings, item_embeddings
        
        
