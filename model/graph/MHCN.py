# Torch Version
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
# paper: Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation. WWW'21

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

class MHCN(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(MHCN, self).__init__(conf, training_set, test_set, **kwargs)
        args = OptionConf(self.config['MHCN'])
        self.n_layers = int(args['-n_layer'])
        self.ss_rate = float(args['-ss_rate'])
        self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
        self.hypergraphs = self.build_hyper_adj_mats()
        self.model = MHCN_Encoder(self.data, self.hypergraphs, self.emb_size, self.n_layers)

    def print_model_info(self):
        super(MHCN, self).print_model_info()
        # # print social relation statistics
        print('Social data size: (user number: %d, relation number: %d).' % (self.social_data.size()))
        print('=' * 80)

    def build_hyper_adj_mats(self):
        S = self.social_data.get_social_mat()
        Y = self.data.interaction_mat
        B = S.multiply(S.T)
        U = S - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
        A5 = C5 + C5.T
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
        A8 = (Y.dot(Y.T)).multiply(B)
        A9 = (Y.dot(Y.T)).multiply(U)
        A9 = A9 + A9.T
        A10 = Y.dot(Y.T) - A8 - A9
        # addition and row-normalization
        H_s = sum([A1,A2,A3,A4,A5,A6,A7])
        H_s = H_s.multiply(1.0/(H_s.sum(axis=1)+1e-8).reshape(-1, 1))
        H_j = sum([A8,A9])
        H_j = H_j.multiply(1.0/(H_j.sum(axis=1)+1e-8).reshape(-1, 1))
        H_p = A10
        H_p = H_p.multiply(H_p>1)
        H_p = H_p.multiply(1.0/(H_p.sum(axis=1)+1e-8).reshape(-1, 1))

        H_s = TorchGraphInterface.convert_sparse_mat_to_tensor(H_p).cuda()
        H_j = TorchGraphInterface.convert_sparse_mat_to_tensor(H_j).cuda()
        H_p = TorchGraphInterface.convert_sparse_mat_to_tensor(H_p).cuda()
        return [H_s, H_j, H_p]
        
    def hierarchical_self_supervision(self, em, adj):
        def row_shuffle(embedding):
            return embedding[torch.randperm(embedding.size(0))]
        def row_column_shuffle(embedding):
            column_corrupted = embedding[:, torch.randperm(embedding.size(1))]
            return column_corrupted[torch.randperm(embedding.size(0))]
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), dim=1)

        user_embeddings = em
        # For Douban, normalization is needed. (But actually, worse)
        #user_embeddings = nn.functional.normalize(user_embeddings, p=2, dim=1)
        edge_embeddings = torch.sparse.mm(adj, user_embeddings)
        #local MIM
        pos = score(user_embeddings, edge_embeddings)
        neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
        neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
        local_loss = torch.sum(-torch.log(torch.sigmoid(pos-neg1))-torch.log(torch.sigmoid(neg1-neg2)))
        # global mim
        graph = torch.mean(edge_embeddings, dim=0, keepdim=True)
        pos = score(edge_embeddings, graph)
        neg1 = score(row_column_shuffle(edge_embeddings), graph)
        global_loss = torch.sum(-torch.log(torch.sigmoid(pos-neg1)))
        return global_loss + local_loss

    def ss_loss(self, sgating_user_emb, hypergraphs):
        ss_loss = 0.
        ss_loss += self.hierarchical_self_supervision(sgating_user_emb[0], hypergraphs[0])
        ss_loss += self.hierarchical_self_supervision(sgating_user_emb[1], hypergraphs[1])
        ss_loss += self.hierarchical_self_supervision(sgating_user_emb[2], hypergraphs[2])
        return ss_loss
    
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        while not early_stopping:
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, sgating_user_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                ss_loss = self.ss_rate * self.ss_loss(sgating_user_emb, self.hypergraphs) / self.batch_size
                # Backward and optimize
                optimizer.zero_grad()
                total_loss = batch_loss + ss_loss
                total_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item(), 'ss_loss:', ss_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb, _ = model()
            _, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")
    
    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb, _ = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class MHCN_Encoder(nn.Module):
    def __init__(self, data, hypergraphs, emb_size, n_layers):
        super(MHCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.n_channel = 4
        self.norm_inter = data.norm_inter
        self.embedding_dict = self._init_model()
        self.sparse_norm_inter = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_inter).cuda()
        self.H_s = hypergraphs[0]
        self.H_j = hypergraphs[1]
        self.H_p = hypergraphs[2]
    
    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = {}
        for i in range(self.n_channel):
            embedding_dict['gating%d'%i] = nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size)))
            embedding_dict['gating_bias%d'%i] = nn.Parameter(initializer(torch.empty(1, self.latent_size)))
            embedding_dict['sgating%d'%i] = nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size)))
            embedding_dict['sgating_bias%d'%i] = nn.Parameter(initializer(torch.empty(1, self.latent_size)))
    
        embedding_dict['att_mat'] = nn.Parameter(initializer(torch.empty(self.latent_size, self.latent_size)))
        embedding_dict['att_agg'] = nn.Parameter(initializer(torch.empty(1, self.latent_size)))
        embedding_dict['user_emb'] = nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size)))
        embedding_dict['item_emb'] = nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size)))
        embedding_dict = nn.ParameterDict(embedding_dict)
        return embedding_dict

    def self_gating(self, em, channel):
        return torch.mul(em, torch.sigmoid(torch.matmul(em, self.embedding_dict['gating%d'%channel]) + self.embedding_dict['gating_bias%d'%channel]))

    def self_supervised_gating(self, em, channel):
        return torch.mul(em, torch.sigmoid(torch.matmul(em, self.embedding_dict['sgating%d'%channel])+self.embedding_dict['sgating_bias%d'%channel]))

    def channel_attention(self, channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(torch.sum(torch.mul(self.embedding_dict['att_agg'], torch.matmul(embedding, self.embedding_dict['att_mat'])), 1))
        weights = torch.stack(weights)
        score = torch.split(torch.softmax(weights, 0), [1, 1, 1], dim=0)
        mixed_embeddings = 0.
        for i in range(len(score)):
            mixed_embeddings += torch.mul(score[i], channel_embeddings[i].transpose(0, 1)).transpose(0, 1)
        return mixed_embeddings, score
    
    def forward(self):
        #self-gating units
        user_embeddings = self.embedding_dict['user_emb']
        item_embeddings = self.embedding_dict['item_emb']
        
        user_embeddings_c0 = self.self_gating(user_embeddings, 0)
        user_embeddings_c1 = self.self_gating(user_embeddings, 1)
        user_embeddings_c2 = self.self_gating(user_embeddings, 2)
        user_embeddings_simple = self.self_gating(user_embeddings, 3)
        all_embeddings_c0 = [user_embeddings_c0]
        all_embeddings_c1 = [user_embeddings_c1]
        all_embeddings_c2 = [user_embeddings_c2]
        all_embeddings_simple = [user_embeddings_simple]
        all_item_embeddings = [item_embeddings]

        for k in range(self.layers):
            mixed_embeddings, scores = self.channel_attention([user_embeddings_c0, user_embeddings_c1, user_embeddings_c2])
            mixed_embeddings = (mixed_embeddings + user_embeddings_simple) / 2

            # channel s
            user_embeddings_c0 = torch.sparse.mm(self.H_s, user_embeddings_c0)
            norm_embeddings = nn.functional.normalize(user_embeddings_c0, p=2, dim=1)
            all_embeddings_c0 += [norm_embeddings]
            # channel j
            user_embeddings_c1 = torch.sparse.mm(self.H_j, user_embeddings_c1)
            norm_embeddings = nn.functional.normalize(user_embeddings_c1, p=2, dim=1)
            all_embeddings_c1 += [norm_embeddings]
            # channel p
            user_embeddings_c2 = torch.sparse.mm(self.H_p, user_embeddings_c2)
            norm_embeddings = nn.functional.normalize(user_embeddings_c2, p=2, dim=1)
            all_embeddings_c2 += [norm_embeddings]
            # item convolution
            new_item_embeddings = torch.sparse.mm(self.sparse_norm_inter.transpose(0, 1), mixed_embeddings)
            norm_embeddings = nn.functional.normalize(new_item_embeddings, p=2, dim=1)
            all_item_embeddings += [norm_embeddings]
            user_embeddings_simple = torch.sparse.mm(self.sparse_norm_inter, item_embeddings)
            norm_embeddings = nn.functional.normalize(user_embeddings_simple, p=2, dim=1)
            all_embeddings_simple += [norm_embeddings]
            item_embeddings = new_item_embeddings

        # average channel-specific embeddings
        user_embeddings_c0 = torch.stack(all_embeddings_c0, dim=0).sum(dim=0)
        user_embeddings_c1 = torch.stack(all_embeddings_c1, dim=0).sum(dim=0)
        user_embeddings_c2 = torch.stack(all_embeddings_c2, dim=0).sum(dim=0)
        user_embeddings_simple = torch.stack(all_embeddings_simple, dim=0).sum(dim=0)
        item_embeddings = torch.stack(all_item_embeddings, dim=0).sum(dim=0)

        # aggregating channel-specific embeddings
        self.final_item_embeddings = item_embeddings
        self.final_user_embeddings, self.attention_score = self.channel_attention([user_embeddings_c0, user_embeddings_c1, user_embeddings_c2])
        self.final_user_embeddings += user_embeddings_simple / 2

        self.sgating_user_embeddings_c0 = self.self_supervised_gating(self.final_user_embeddings, 0)
        self.sgating_user_embeddings_c1 = self.self_supervised_gating(self.final_user_embeddings, 1)
        self.sgating_user_embeddings_c2 = self.self_supervised_gating(self.final_user_embeddings, 2)

        return self.final_user_embeddings, self.final_item_embeddings, [self.sgating_user_embeddings_c0, self.sgating_user_embeddings_c1, self.sgating_user_embeddings_c2]


# Tensorflow Version
# =============================================================================
# from base.graph_recommender import GraphRecommender
# import tensorflow as tf
# from util.loss_tf import bpr_loss
# from data.social import Relation
# from base.tf_interface import TFGraphInterface
# from util.sampler import next_batch_pairwise
# from util.conf import OptionConf
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# 
# # paper: Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation. WWW'21
# 
# 
# class MHCN(GraphRecommender):
#     def __init__(self, conf, training_set, test_set, **kwargs):
#         GraphRecommender.__init__(self, conf, training_set, test_set, **kwargs)
#         args = OptionConf(self.config['MHCN'])
#         self.n_layers = int(args['-n_layer'])
#         self.ss_rate = float(args['-ss_rate'])
#         self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
# 
#     def print_model_info(self):
#         super(MHCN, self).print_model_info()
#         # # print social relation statistics
#         print('Social data size: (user number: %d, relation number: %d).' % (self.social_data.size()))
#         print('=' * 80)
# 
#     def build_hyper_adj_mats(self):
#         S = self.social_data.get_social_mat()
#         Y = self.data.interaction_mat
#         B = S.multiply(S.T)
#         U = S - B
#         C1 = (U.dot(U)).multiply(U.T)
#         A1 = C1 + C1.T
#         C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
#         A2 = C2 + C2.T
#         C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
#         A3 = C3 + C3.T
#         A4 = (B.dot(B)).multiply(B)
#         C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
#         A5 = C5 + C5.T
#         A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
#         A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
#         A8 = (Y.dot(Y.T)).multiply(B)
#         A9 = (Y.dot(Y.T)).multiply(U)
#         A9 = A9 + A9.T
#         A10 = Y.dot(Y.T) - A8 - A9
#         # addition and row-normalization
#         H_s = sum([A1,A2,A3,A4,A5,A6,A7])
#         H_s = H_s.multiply(1.0/H_s.sum(axis=1).reshape(-1, 1))
#         H_j = sum([A8,A9])
#         H_j = H_j.multiply(1.0/H_j.sum(axis=1).reshape(-1, 1))
#         H_p = A10
#         H_p = H_p.multiply(H_p>3)
#         H_p = H_p.multiply(1.0/H_p.sum(axis=1).reshape(-1, 1))
#         return [H_s, H_j, H_p]
# 
#     def build(self):
#         self.weights = {}
#         self.n_channel = 4
#         initializer = tf.contrib.layers.xavier_initializer()
#         self.user_embeddings = tf.Variable(initializer([self.data.user_num, self.emb_size]))
#         self.item_embeddings = tf.Variable(initializer([self.data.item_num, self.emb_size]))
#         self.u_idx = tf.placeholder(tf.int32, name="u_idx")
#         self.v_idx = tf.placeholder(tf.int32, name="v_idx")
#         self.neg_idx = tf.placeholder(tf.int32, name="neg_holder")
#         # define learnable paramters
#         for i in range(self.n_channel):
#             self.weights['gating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='g_W_%d_1' % (i + 1))
#             self.weights['gating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]), name='g_W_b_%d_1' % (i + 1))
#             self.weights['sgating%d' % (i + 1)] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='sg_W_%d_1' % (i + 1))
#             self.weights['sgating_bias%d' % (i + 1)] = tf.Variable(initializer([1, self.emb_size]), name='sg_W_b_%d_1' % (i + 1))
#         self.weights['attention'] = tf.Variable(initializer([1, self.emb_size]), name='at')
#         self.weights['attention_mat'] = tf.Variable(initializer([self.emb_size, self.emb_size]), name='atm')
#         tf_config = tf.ConfigProto()
#         tf_config.gpu_options.allow_growth = True
#         self.sess = tf.Session(config=tf_config)
# 
#         # define inline functions
#         def self_gating(em, channel):
#             return tf.multiply(em, tf.nn.sigmoid(tf.matmul(em, self.weights['gating%d' % channel]) + self.weights['gating_bias%d' % channel]))
# 
#         def self_supervised_gating(em, channel):
#             return tf.multiply(em, tf.nn.sigmoid(tf.matmul(em, self.weights['sgating%d' % channel]) + self.weights['sgating_bias%d' % channel]))
# 
#         def channel_attention(*channel_embeddings):
#             weights = []
#             for embedding in channel_embeddings:
#                 weights.append(tf.reduce_sum(tf.multiply(self.weights['attention'], tf.matmul(embedding, self.weights['attention_mat'])), 1))
#             score = tf.nn.softmax(tf.transpose(weights))
#             mixed_embeddings = 0
#             for i in range(len(weights)):
#                 mixed_embeddings += tf.transpose(tf.multiply(tf.transpose(score)[i], tf.transpose(channel_embeddings[i])))
#             return mixed_embeddings, score
# 
#         # initialize adjacency matrices
#         M_matrices = self.build_hyper_adj_mats()
#         H_s = M_matrices[0]
#         H_s = TFGraphInterface.convert_sparse_mat_to_tensor(H_s)
#         H_j = M_matrices[1]
#         H_j = TFGraphInterface.convert_sparse_mat_to_tensor(H_j)
#         H_p = M_matrices[2]
#         H_p = TFGraphInterface.convert_sparse_mat_to_tensor(H_p)
#         R = TFGraphInterface.convert_sparse_mat_to_tensor(self.data.normalize_graph_mat(self.data.interaction_mat))
#         # self-gating
#         user_embeddings_c1 = self_gating(self.user_embeddings, 1)
#         user_embeddings_c2 = self_gating(self.user_embeddings, 2)
#         user_embeddings_c3 = self_gating(self.user_embeddings, 3)
#         simple_user_embeddings = self_gating(self.user_embeddings, 4)
#         all_embeddings_c1 = [user_embeddings_c1]
#         all_embeddings_c2 = [user_embeddings_c2]
#         all_embeddings_c3 = [user_embeddings_c3]
#         all_embeddings_simple = [simple_user_embeddings]
#         item_embeddings = self.item_embeddings
#         all_embeddings_i = [item_embeddings]
# 
#         self.ss_loss = 0
#         # multi-channel convolution
#         for k in range(self.n_layers):
#             mixed_embedding = channel_attention(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)[0] + simple_user_embeddings / 2
#             # Channel S
#             user_embeddings_c1 = tf.sparse_tensor_dense_matmul(H_s, user_embeddings_c1)
#             norm_embeddings = tf.math.l2_normalize(user_embeddings_c1, axis=1)
#             all_embeddings_c1 += [norm_embeddings]
#             # Channel J
#             user_embeddings_c2 = tf.sparse_tensor_dense_matmul(H_j, user_embeddings_c2)
#             norm_embeddings = tf.math.l2_normalize(user_embeddings_c2, axis=1)
#             all_embeddings_c2 += [norm_embeddings]
#             # Channel P
#             user_embeddings_c3 = tf.sparse_tensor_dense_matmul(H_p, user_embeddings_c3)
#             norm_embeddings = tf.math.l2_normalize(user_embeddings_c3, axis=1)
#             all_embeddings_c3 += [norm_embeddings]
#             # item convolution
#             new_item_embeddings = tf.sparse_tensor_dense_matmul(tf.sparse.transpose(R), mixed_embedding)
#             norm_embeddings = tf.math.l2_normalize(new_item_embeddings, axis=1)
#             all_embeddings_i += [norm_embeddings]
#             simple_user_embeddings = tf.sparse_tensor_dense_matmul(R, item_embeddings)
#             all_embeddings_simple += [tf.math.l2_normalize(simple_user_embeddings, axis=1)]
#             item_embeddings = new_item_embeddings
#         # averaging the channel-specific embeddings
#         user_embeddings_c1 = tf.reduce_sum(all_embeddings_c1, axis=0)
#         user_embeddings_c2 = tf.reduce_sum(all_embeddings_c2, axis=0)
#         user_embeddings_c3 = tf.reduce_sum(all_embeddings_c3, axis=0)
#         simple_user_embeddings = tf.reduce_sum(all_embeddings_simple, axis=0)
#         item_embeddings = tf.reduce_sum(all_embeddings_i, axis=0)
#         # aggregating channel-specific embeddings
#         self.final_item_embeddings = item_embeddings
#         self.final_user_embeddings, self.attention_score = channel_attention(user_embeddings_c1, user_embeddings_c2, user_embeddings_c3)
#         self.final_user_embeddings += simple_user_embeddings / 2
#         # create self-supervised loss
#         self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings, 1), H_s)
#         self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings, 2), H_j)
#         self.ss_loss += self.hierarchical_self_supervision(self_supervised_gating(self.final_user_embeddings, 3), H_p)
#         self.ss_loss = self.ss_rate * self.ss_loss
#         # embedding look-up
#         self.batch_neg_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.neg_idx)
#         self.batch_user_emb = tf.nn.embedding_lookup(self.final_user_embeddings, self.u_idx)
#         self.batch_pos_item_emb = tf.nn.embedding_lookup(self.final_item_embeddings, self.v_idx)
# 
#     def hierarchical_self_supervision(self, em, adj):
#         def row_shuffle(embedding):
#             return tf.gather(embedding, tf.random.shuffle(tf.range(tf.shape(embedding)[0])))
#         def row_column_shuffle(embedding):
#             corrupted_embedding = tf.transpose(tf.gather(tf.transpose(embedding), tf.random.shuffle(tf.range(tf.shape(tf.transpose(embedding))[0]))))
#             corrupted_embedding = tf.gather(corrupted_embedding, tf.random.shuffle(tf.range(tf.shape(corrupted_embedding)[0])))
#             return corrupted_embedding
#         def score(x1, x2):
#             return tf.reduce_sum(tf.multiply(x1, x2), 1)
#         user_embeddings = em
#         #user_embeddings = tf.math.l2_normalize(em,1)
#         edge_embeddings = tf.sparse_tensor_dense_matmul(adj, user_embeddings)
#         # Local MIM
#         pos = score(user_embeddings, edge_embeddings)
#         neg1 = score(row_shuffle(user_embeddings), edge_embeddings)
#         neg2 = score(row_column_shuffle(edge_embeddings), user_embeddings)
#         local_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos - neg1)) - tf.log(tf.sigmoid(neg1 - neg2)))
#         # Global MIM
#         graph = tf.reduce_mean(edge_embeddings, 0)
#         pos = score(edge_embeddings, graph)
#         neg1 = score(row_column_shuffle(edge_embeddings), graph)
#         global_loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos - neg1)))
#         return global_loss + local_loss
# 
#     def train(self):
#         rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
#         reg_loss = 0
#         for key in self.weights:
#             reg_loss += self.reg * tf.nn.l2_loss(self.weights[key])
#         reg_loss += self.reg * (tf.nn.l2_loss(self.batch_user_emb) + tf.nn.l2_loss(self.batch_neg_item_emb) + tf.nn.l2_loss(self.batch_pos_item_emb))
#         total_loss = rec_loss + reg_loss + self.ss_loss
#         opt = tf.train.AdamOptimizer(self.lRate)
#         train_op = opt.minimize(total_loss)
#         init = tf.global_variables_initializer()
#         self.sess.run(init)
#         # Suggested Maximum epoch Setting: LastFM 120 Douban 30 Yelp 30
#         for epoch in range(self.maxEpoch):
#             for n, batch in enumerate(next_batch_pairwise(self.data,self.batch_size)):
#                 user_idx, i_idx, j_idx = batch
#                 _, l1, l2 = self.sess.run([train_op, rec_loss, self.ss_loss], feed_dict={self.u_idx: user_idx, self.neg_idx: j_idx, self.v_idx: i_idx})
#                 print('training:', epoch + 1, 'batch', n, 'rec loss:',l1,'ssl loss',l2)
#             self.U, self.V = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])
#             self.fast_evaluation(epoch)
#         self.U, self.V = self.best_user_emb, self.best_item_emb
# 
#     def save(self):
#         self.best_user_emb, self.best_item_emb = self.sess.run([self.final_user_embeddings, self.final_item_embeddings])
# 
#     def predict(self, u):
#         u = self.data.get_user_id(u)
#         return self.V.dot(self.U[u])
# 
# 
# =============================================================================
