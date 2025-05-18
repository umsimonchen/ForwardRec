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
import math
# paper: CDAE: Collaborative denoising auto-encoders for top-N recommender systems. WSDM'16
# https://github.com/jasonyaw/CDAE

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

class CDAE(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(CDAE, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['CDAE'])
        self.corruption_ratio = float(args['-corruption_ratio'])
        self.model = CDAE_Encoder(self.data, self.emb_size, self.corruption_ratio)
        self.norm_inter = self.data.norm_inter
    
    def next_batch(self):
        X = np.zeros((self.batch_size,self.data.item_num))
        uids = []
        positive = np.zeros((self.batch_size, self.data.item_num))
        negative = np.zeros((self.batch_size, self.data.item_num))
        userList = list(self.data.user.keys())
        itemList = list(self.data.item.keys())
        for n in range(self.batch_size):
            user = random.choice(userList)
            uids.append(self.data.user[user])
            vec = self.data.row(self.data.user[user]).astype(bool).astype(float)
            ratedItems = self.data.training_set_u[user]
            for item in ratedItems:
                iid = self.data.item[item]
                positive[n][iid]=1
            for i in range(1*len(ratedItems)):
                ng = random.choice(itemList)
                while ng in self.data.training_set_u[user]:
                    ng = random.choice(itemList)
                n_id = self.data.item[ng]
                negative[n][n_id]=1
            #X[n]=self.norm_inter[self.data.user[user]].toarray()[0]
            X[n] = vec
        return X,uids,positive,negative
    
    def given_batch(self,uids):
        user_len = len(uids)
        X = np.zeros((user_len,self.data.item_num))
        positive = np.zeros((user_len, self.data.item_num))
        negative = np.zeros((user_len, self.data.item_num))
        userList = list(self.data.user.keys())
        itemList = list(self.data.item.keys())
        for n in range(user_len):
            vec = self.data.row(uids[n]).astype(bool).astype(float)
            ratedItems = self.data.training_set_u[self.data.id2user[uids[n]]]
            for item in ratedItems:
                iid = self.data.item[item]
                positive[n][iid]=1
            for i in range(1*len(ratedItems)):
                ng = random.choice(itemList)
                while ng in self.data.training_set_u[self.data.id2user[uids[n]]]:
                    ng = random.choice(itemList)
                n_id = self.data.item[ng]
                negative[n][n_id]=1
            #X[n]=self.norm_inter[self.data.user[user]].toarray()[0]
            X[n] = vec
        return X,uids,positive,negative
    
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        while not early_stopping:
            batch_xs, user, positive, negative = self.next_batch()
            mask = torch.tensor(np.random.binomial(1, self.corruption_ratio,(self.batch_size, self.data.item_num)), dtype=torch.float).cuda()
            y_pred, y_positive, y_negative, _ = model(batch_xs, user, positive, negative, mask)
            loss = -torch.mul(y_positive, torch.log(y_pred))-torch.mul(y_negative, torch.log(1-y_pred))
            reg_loss = l2_reg_loss(self.reg, model.embedding_dict['user_emb'], model.weight_dict['encoder_w'], model.weight_dict['encoder_b'], model.weight_dict['decoder_w'], model.weight_dict['decoder_b'])
            total_loss = torch.mean(loss) + reg_loss#/self.batch_size
            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print('training:', epoch + 1, 'total_loss:', total_loss.item())
            with torch.no_grad():
                self.scores = []
                total_batch = math.ceil(self.data.user_num/self.batch_size)
                for i in range(total_batch):
                    user_idx = list(range(i*self.batch_size, min(self.data.user_num,(i+1)*self.batch_size)))                
                    batch_xs, user, positive, negative = self.given_batch(user_idx)
                    mask = torch.tensor(np.ones((1,self.data.item_num)), dtype=torch.float).cuda()
                    _, _, _, decoder_op = model(batch_xs, user, positive, negative, mask)
                    self.scores.append(decoder_op.cpu().numpy())
                    print("Finished evaluation %d / %d."%(i+1,total_batch))
                self.scores = np.concatenate(self.scores, axis=0)
            measure, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.scores = self.best_scores
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")
        # record training loss        
        # with open('training_record','wb') as fp:
        #     pickle.dump([record_list, loss_list], fp)
        
    def save(self):
        with torch.no_grad():
            self.best_scores = self.scores

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = self.scores[u]
        return score

class CDAE_Encoder(nn.Module):
    def __init__(self, data, emb_size, corruption_ratio):
        super(CDAE_Encoder, self).__init__()
        self.data = data
        self.embedding_size = emb_size
        self.corruption_ratio = corruption_ratio
        self.embedding_dict, self.weight_dict = self._init_model()
       
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.embedding_size))),
        })
        weight_dict = nn.ParameterDict({
            'encoder_w': nn.Parameter(initializer(torch.empty(self.data.item_num, self.embedding_size))),
            'decoder_w': nn.Parameter(initializer(torch.empty(self.embedding_size, self.data.item_num))),
            'encoder_b': nn.Parameter(initializer(torch.empty(1,self.embedding_size))),
            'decoder_b': nn.Parameter(initializer(torch.empty(1,self.data.item_num)))
        })
        return embedding_dict, weight_dict

    def encoder(self, x, v):
        layer = nn.Sigmoid()(torch.matmul(x, self.weight_dict['encoder_w'])+self.weight_dict['encoder_b']+v)
        return layer

    def decoder(self, x):
        layer = nn.Sigmoid()(torch.matmul(x, self.weight_dict['decoder_w'])+self.weight_dict['decoder_b'])
        return layer
    
    def forward(self, X, user, positive, negative, mask):
        X = torch.tensor(X, dtype=torch.float).cuda()
        user = torch.tensor(user).cuda()
        positive = torch.tensor(positive).cuda()
        negative = torch.tensor(negative).cuda()
        
        corrupted_input = torch.mul(mask, X)
        encoder_op = self.encoder(corrupted_input, self.embedding_dict['user_emb'][user])
        decoder_op = self.decoder(encoder_op)
        y_pred = torch.mul(decoder_op, mask)
        y_pred = torch.clamp(y_pred, min=1e-6)
        y_positive = torch.mul(positive, mask)
        y_negative = torch.mul(negative, mask)
        return y_pred, y_positive, y_negative, decoder_op 


