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

class ForwardRec(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(ForwardRec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['ForwardRec'])
        self.n_layers = int(args['-n_layer'])
        self.neg_factor = float(args['-neg_factor'])
        self.model = ForwardRec_Encoder(self.data, self.emb_size, self.n_layers)
		# not purchased, we need
        self.unobserved_adj = torch.logical_not(TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda().to_dense().to(torch.float32))
        #self.observed_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat).cuda().to_dense().to(torch.float32)
        
    def train(self):
        record_list = []
        loss_list = []
        model = self.model.cuda()
        all_performances = []
        for self.current_layer in range(self.n_layers):
            early_stopping = False
            flag = True
            last_performance = []
            epoch = 0
            best_epoch = 0
			# Refresh the best performance
            self.bestPerformance = []
            
			# Layer normalization
            if self.current_layer != 0:
                model.embedding_dict['user_emb'] = nn.functional.normalize(model.embedding_dict['user_emb'], p=2, dim=1)
                model.embedding_dict['item_emb'] = nn.functional.normalize(model.embedding_dict['item_emb'], p=2, dim=1)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
            while not early_stopping:
                for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                    user_idx, pos_idx, random_neg_idx = batch
                    rec_user_emb, rec_item_emb = model()
                    
                    #user_temp = rec_user_emb.clone().detach()
                    #item_temp = rec_item_emb.clone().detach()
					
					# negative sample generation w/ neg_factor
                    ui_score = torch.matmul(rec_user_emb[user_idx], rec_item_emb.transpose(0, 1))
                    _, indices = torch.sort(ui_score, dim=1, descending=True, stable=True)
                    neg_idx = torch.zeros_like(ui_score, dtype=torch.float32)
                    neg_idx.scatter_(1, indices[:,:int(self.neg_factor*self.data.item_num)], 1.0)
                    neg_idx = torch.logical_and(neg_idx, self.unobserved_adj[user_idx]).to(torch.float)
                    _, neg_idx = torch.sort(torch.mul(torch.randn_like(neg_idx), neg_idx), dim=1, descending=True, stable=True)
                    neg_idx = neg_idx[:,0]
					
					# negative sample generation w/o neg_factor
					# masked = self.unobserved_adj.to(torch.float)[user_idx]
					# ui_score = torch.matmul(user_temp[user_idx], item_temp.transpose(0, 1))
					# ui_score = torch.mul(masked, ui_score) # masked observed items
					# values, indices = torch.sort(ui_score, dim=1, descending=True, stable=True)
					# min_pos = torch.min((values>0).sum(dim=1, keepdim=True))
					# y_all = torch.log(values[:,:min_pos])
					
					# x_all = torch.arange(min_pos).unsqueeze(0).repeat(len(user_idx),1).to(torch.float).cuda()
					# xmean_all = torch.tensor((min_pos-1)/2, dtype=torch.float).cuda()
					# ymean_all = y_all.mean(dim=1, keepdim=True)
					# k_all = torch.mul((x_all-xmean_all), (y_all-ymean_all)).sum(dim=1, keepdim=True) / torch.square(x_all-xmean_all).sum(dim=1, keepdim=True)
					# b_all = ymean_all - k_all*xmean_all
					# ydiff_all = y_all - torch.mul(x_all, k_all) - b_all
					# ydiffmean_all = ydiff_all.mean(dim=1, keepdim=True)
					# ydiffvar_all = torch.square(ydiffmean_all - ydiff_all).sum(dim=1, keepdim=True) / min_pos
					
					# x_top = torch.arange(self.max_N).unsqueeze(0).repeat(len(user_idx),1).cuda()
					# y_top = y_all[:, :self.max_N]
					# ymean_top = y_top.mean(dim=1, keepdim=True)
					# xmean_top = torch.tensor((self.max_N-1)/2, dtype=torch.float).cuda()
					# k_top = torch.mul((x_top-xmean_top), (y_top-ymean_top)).sum(dim=1, keepdim=True) / torch.square(x_top-xmean_top).sum(dim=1, keepdim=True)
					# b_top = ymean_top - k_top*xmean_top
					# ydiff_top = y_top - torch.mul(x_top, k_top) - b_top
					# ydiffmean_top = ydiff_top.mean(dim=1, keepdim=True)
					# ydiffvar_top = torch.square(ydiffmean_top - ydiff_top).sum(dim=1, keepdim=True) / (self.max_N)
					
					# ydiff1 = torch.abs(y_all - torch.mul(k_all, x_all) - b_all)
					# ydiff2 = torch.abs(y_all - torch.mul(k_top, x_all) - b_top)
					# neg = (ydiff1>=2*torch.sqrt(ydiffvar_all)) & (ydiff2>=2*torch.sqrt(ydiffvar_top))
					# neg[:,:self.max_N] = False
					# neg = torch.mul(neg.to(torch.float), torch.rand_like(neg.to(torch.float)))
					# max_idx = torch.argmax(neg, dim=1, keepdim=True)
					# neg_idx = torch.gather(indices, 1, max_idx).squeeze(1)
					
					# Backward and optimize
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                    batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    if n%100 == 0 and n > 0:
                        print('layer:', self.current_layer, 'training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
                        
				# Validation
                with torch.no_grad():
                    print("\nLayer %d:"%self.current_layer)
                    self.user_emb, self.item_emb = model()
                    measure, early_stopping = self.fast_evaluation(epoch)
                    record_list.append(measure)
                    
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[random_neg_idx]
                    batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                    loss_list.append(batch_loss.item())
				
				# Checking in case of huge gap after normalization
                if flag:
                    if len(last_performance)==0:
                        last_performance = measure
                    else:
                        count = 0
                        for i in range(4):
                            if float(last_performance[i].split(':')[1]) > float(measure[i].split(':')[1]):
                                count+=1
                            else:
                                count-=1
                        if count>0:
                            self.bestPerformance[0] = epoch+1
                            self.bestPerformance[1]['Hit Ratio'] = float(measure[0].split(':')[1])
                            self.bestPerformance[1]['Precision'] = float(measure[1].split(':')[1])
                            self.bestPerformance[1]['Recall'] = float(measure[2].split(':')[1])
                            self.bestPerformance[1]['NDCG'] = float(measure[3].split(':')[1])
                            last_performance = measure
                        else:
                            flag = False
					
                epoch += 1
                best_epoch = self.bestPerformance[0]
            all_performances.append(self.bestPerformance[1])
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        # record performance
        with open('performance.txt','a') as fp:
            for n, performance in enumerate(all_performances):
                fp.write("At layer %d"%n + str(performance)+"\n")
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


class ForwardRec_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(ForwardRec_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size)))
        })
        return embedding_dict

    def forward(self):
        user_emb = self.embedding_dict['user_emb']
        item_emb = self.embedding_dict['item_emb']
        ego_embeddings = torch.cat([user_emb, item_emb], dim=0)
        all_embeddings = [ego_embeddings]
        ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
        all_embeddings += [ego_embeddings]
        #all_embeddings = torch.stack(all_embeddings, dim=1)
        #all_embeddings = torch.mean(all_embeddings, dim=1)
        all_embeddings = all_embeddings[-1] #max
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


