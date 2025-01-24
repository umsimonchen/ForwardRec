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
import scipy as sp
import scipy.sparse.linalg
import pickle
# paper: PGSP: Personalized Graph Signal Processing for Collaborative Filtering. WWW'23
# https://github.com/jhliu0807/PGSP/blob/main/PGSP.ipynb

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

class PGSP(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(PGSP, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['PGSP'])
        self.n_layers = int(args['-n_layer'])
        self.frequency = int(args['-frequency'])
        self.phi = float(args['-phi'])
        self.model = PGSP_Encoder(self.data, self.n_layers, self.frequency, self.phi)

    def train(self):
        model = self.model.cuda()
        self.P = model()
        _, _ = self.fast_evaluation(0)
        self.P = self.best_P
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")

    def save(self):
        with torch.no_grad():
            self.best_P = self.P

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = self.P[u]
        return score

class PGSP_Encoder(nn.Module):
    def __init__(self, data, n_layers, frequency, phi):
        super(PGSP_Encoder, self).__init__()
        self.data = data
        self.layers = n_layers
        self.frequency = frequency
        self.phi = phi 

    def forward(self):
        # construct specific adjacency matrix
        R = self.data.interaction_mat
        Du_ = sp.sparse.diags(np.power(R.sum(axis=1).T.A[0], -1/2), offsets=0)
        Di_ = sp.sparse.diags(np.power(R.sum(axis=0).A[0], -1/2), offsets=0)
        Du = sp.sparse.diags(np.power(R.sum(axis=1).T.A[0], 1/2), offsets=0)
        Di = sp.sparse.diags(np.power(R.sum(axis=0).A[0], 1/2), offsets=0)
        Ru = Du_ * R 
        Ri = R * Di_
        Cu = Ri * Ri.T
        Ci = Ru.T * Ru
        R_post = Ru * Di_
        Ci0 = R_post.T * R_post
        Cu0 = R_post * R_post.T
        A = sp.sparse.vstack([sp.sparse.hstack([Cu, R]), sp.sparse.hstack([R.T, Ci])])
        I = sp.sparse.identity(self.data.user_num + self.data.item_num, dtype=np.float32)
        D_ = sp.sparse.diags(np.power(A.sum(axis=0).A[0], -1/2), offsets=0)
        A_norm = D_ * A * D_
        L_norm = I - A_norm

        try:
            with open('dataset/gowalla/PGSP', 'rb') as fp: # choose the dataset folder
                eigen = pickle.load(fp)
            val = eigen[0]
            vec = eigen[1]
        except:
            val, vec = sp.sparse.linalg.eigsh(L_norm, k=self.frequency, which='SA', tol=0)
            with open('dataset/gowalla/PGSP', 'wb') as fp:
                pickle.dump([val, vec], fp)

        R_b = sp.sparse.hstack([Cu0, R])
        D_Rb_i_ = sp.sparse.diags(np.power(R_b.sum(axis=0).A[0], -1/2), offsets=0)
        D_Rb_i = sp.sparse.diags(np.power(R_b.sum(axis=0).A[0], 1/2), offsets=0)
        D_Rb_i = D_Rb_i.toarray()

        P0 = R * Di_
        P0 = P0 * Ci0
        P0 = P0 * Di
        P0 = P0.toarray()

        P1 = R_b * D_Rb_i_
        P1 = P1 * vec
        P11 = np.matmul(vec.T, D_Rb_i)
        P1 = np.matmul(P1, P11)
        P1 = P1[:, self.data.user_num:]
        P = self.phi * P0 + (1 - self.phi) * P1
        return P


