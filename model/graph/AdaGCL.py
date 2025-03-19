import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss, InfoNCE
import os
import numpy as np
import random
import pickle
from copy import deepcopy
import torch_sparse
# paper: AdaGCL: Adaptive Graph Contrastive Learning for Recommendation, KDD'23
# https://github.com/HKUDS/AdaGCL

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

class vgae_encoder(nn.Module):
    def __init__(self, latdim, model):
        super(vgae_encoder, self).__init__()
        hidden = latdim
        self.model = model
        self.encoder_mean = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
        self.encoder_std = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.Softplus())

    def forward(self, adj):
        x = self.model.forward_graphcl(adj)
        x_mean = self.encoder_mean(x)
        x_std = self.encoder_std(x)
        gaussian_noise = torch.randn(x_mean.shape).cuda()
        x = gaussian_noise * x_std + x_mean
        return x, x_mean, x_std

class vgae_decoder(nn.Module):
    def __init__(self, latdim, model):
        super(vgae_decoder, self).__init__()
        hidden = latdim
        self.model = model
        self.decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction='none')
        self.reg = 1e-5 # default constant

    def calcRegLoss(self, model):
        ret = 0
        for W in model.parameters():
            ret += W.norm(2).square()
        return ret
        
    def forward(self, x, x_mean, x_std, users, items, neg_items, encoder):
        x_user, x_item = torch.split(x, [self.model.data.user_num, self.model.data.item_num], dim=0)

        edge_pos_pred = self.sigmoid(self.decoder(x_user[users] * x_item[items]))
        edge_neg_pred = self.sigmoid(self.decoder(x_user[users] * x_item[neg_items]))

        loss_edge_pos = self.bceloss( edge_pos_pred, torch.ones(edge_pos_pred.shape).cuda() )
        loss_edge_neg = self.bceloss( edge_neg_pred, torch.zeros(edge_neg_pred.shape).cuda() )
        loss_rec = loss_edge_pos + loss_edge_neg

        kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std) - x_mean**2 - x_std**2).sum(dim=1)

        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        bprLoss = bpr_loss(ancEmbeds, posEmbeds, negEmbeds)
        regLoss = self.calcRegLoss(encoder) * self.reg
    
        beta = 0.1
        loss = (loss_rec + beta * kl_divergence.mean() + bprLoss + regLoss).mean()
        
        return loss

class vgae(nn.Module):
    def __init__(self, encoder, decoder):
        super(vgae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data, users, items, neg_items):
        x, x_mean, x_std = self.encoder(data)
        loss = self.decoder(x, x_mean, x_std, users, items, neg_items, self.encoder)
        return loss

    def generate(self, data):
        x, _, _ = self.encoder(data)
        edge_index = data._indices()
        
        edge_pred = self.decoder.sigmoid(self.decoder.decoder(x[edge_index[0]] * x[edge_index[1]]))
        vals = data._values()
        idxs = data._indices()
        edgeNum = vals.size()
        edge_pred = edge_pred[:, 0]
        mask = ((edge_pred + 0.5).floor()).type(torch.bool)
        newVals = vals[mask]
          
        newVals = newVals / (newVals.shape[0] / edgeNum[0])
        newIdxs = idxs[:, mask]
              
        return torch.sparse.FloatTensor(newIdxs, newVals, data.shape)

class DenoisingNet(nn.Module):
    def __init__(self, user_num, item_num, latdim, gcnLayers, features):
        super(DenoisingNet, self).__init__()
        self.gcnLayers = gcnLayers
        self.user_num = user_num
        self.item_num = item_num
        self.features = features

        self.edge_weights = []
        self.nblayers = []
        self.selflayers = []

        self.attentions = []
        self.attentions.append([])
        self.attentions.append([])
        self.gamma = -0.45 # default constant
        self.zeta = 1.05 # default constant
        self.lambda0 = 1e-4 # default constant
        self.reg = 1e-5 # default constant

        hidden = latdim

        self.nblayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.nblayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

        self.selflayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.selflayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

        self.attentions_0 = nn.Sequential(nn.Linear( 2 * hidden, 1))
        self.attentions_1 = nn.Sequential(nn.Linear( 2 * hidden, 1))

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    def get_attention(self, input1, input2, layer=0):
        if layer == 0:
            nb_layer = self.nblayers_0
            selflayer = self.selflayers_0
        if layer == 1:
            nb_layer = self.nblayers_1
            selflayer = self.selflayers_1

        input1 = nb_layer(input1)
        input2 = selflayer(input2)

        input10 = torch.concat([input1, input2], axis=1)

        if layer == 0:
            weight10 = self.attentions_0(input10)
        if layer == 1:
            weight10 = self.attentions_1(input10)
    
        return weight10

    def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
        gamma = self.gamma
        zeta = self.zeta

        if training:
            debug_var = 1e-7
            bias = 0.0
            np_random = np.random.uniform(low=debug_var, high=1.0-debug_var, size=np.shape(log_alpha.cpu().detach().numpy()))
            random_noise = bias + torch.tensor(np_random)
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs.cuda() + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        stretched_values = gate_inputs * (zeta-gamma) +gamma
        cliped = torch.clamp(stretched_values, 0.0, 1.0)
        return cliped.float()

    def generate(self, x, layer=0):
        f1_features = x[self.row, :]
        f2_features = x[self.col, :]

        weight = self.get_attention(f1_features, f2_features, layer)

        mask = self.hard_concrete_sample(weight, training=False)

        mask = torch.squeeze(mask)
        adj = torch.sparse.FloatTensor(self.adj_mat._indices(), mask, self.adj_mat.shape)

        ind = deepcopy(adj._indices())
        row = ind[0, :]
        col = ind[1, :]

        rowsum = torch.sparse.sum(adj, dim=-1).to_dense()
        d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
        d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
        row_inv_sqrt = d_inv_sqrt[row]
        col_inv_sqrt = d_inv_sqrt[col]
        values = torch.mul(adj._values(), row_inv_sqrt)
        values = torch.mul(values, col_inv_sqrt)

        support = torch.sparse.FloatTensor(adj._indices(), values, adj.shape)

        return support

    def l0_norm(self, log_alpha, beta):
        gamma = self.gamma
        zeta = self.zeta
        gamma = torch.tensor(gamma)
        zeta = torch.tensor(zeta)
        reg_per_weight = torch.sigmoid(log_alpha - beta * torch.log(-gamma/zeta))

        return torch.mean(reg_per_weight)

    def set_fea_adj(self, nodes, adj):
        self.node_size = nodes
        self.adj_mat = adj

        ind = deepcopy(adj._indices())

        self.row = ind[0, :]
        self.col = ind[1, :]

    def call(self, inputs, training=None):
        if training:
            temperature = inputs
        else:
            temperature = 1.0

        self.maskes = []

        x = self.features.detach()
        layer_index = 0
        embedsLst = [self.features.detach()]

        for layer in self.gcnLayers:
            xs = []
            f1_features = x[self.row, :]
            f2_features = x[self.col, :]

            weight = self.get_attention(f1_features, f2_features, layer=layer_index)
            mask = self.hard_concrete_sample(weight, temperature, training)

            self.edge_weights.append(weight)
            self.maskes.append(mask)
            mask = torch.squeeze(mask)

            adj = torch.sparse.FloatTensor(self.adj_mat._indices(), mask, self.adj_mat.shape).coalesce()
            ind = deepcopy(adj._indices())
            row = ind[0, :]
            col = ind[1, :]

            rowsum = torch.sparse.sum(adj, dim=-1).to_dense() + 1e-6
            d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
            d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
            row_inv_sqrt = d_inv_sqrt[row]
            col_inv_sqrt = d_inv_sqrt[col]
            values = torch.mul(adj.values(), row_inv_sqrt)
            values = torch.mul(values, col_inv_sqrt)
            support = torch.sparse.FloatTensor(adj._indices(), values, adj.shape).coalesce()
            
            nextx = layer(support, x, False)
            xs.append(nextx)
            x = xs[0]
            embedsLst.append(x)
            layer_index += 1
        return sum(embedsLst)
  
    def lossl0(self, temperature):
        l0_loss = torch.zeros([]).cuda()
        for weight in self.edge_weights:
            l0_loss += self.l0_norm(weight, temperature)
        self.edge_weights = []
        return l0_loss
    
    def calcRegLoss(self):
        ret = 0
        for W in self.parameters():
            ret += W.norm(2).square()
        return ret
    
    def forward(self, users, items, neg_items, temperature):
        self.freeze(self.gcnLayers)
        x = self.call(temperature, True)
        x_user, x_item = torch.split(x, [self.user_num, self.item_num], dim=0)
        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        bprLoss = bpr_loss(ancEmbeds, posEmbeds, negEmbeds)
        regLoss = self.calcRegLoss() * self.reg

        lossl0 = self.lossl0(temperature) * self.lambda0
        return bprLoss + regLoss + lossl0

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds, flag=True):
        if (flag):
            return torch.spmm(adj, embeds)
        else:
            try:
                return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)
            except: # stable for reproducibility
                return torch.sparse.mm(adj, embeds)

class AdaGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(AdaGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['AdaGCL'])
        self.n_layers = int(args['-n_layer'])
        self.ib_reg = float(args['-ib_reg'])
        self.ssl_reg = 0.1 # default constant
        
        self.model = AdaGCL_Encoder(self.data, self.emb_size, self.n_layers)
        
    def train(self):
        model = self.model.cuda()
        early_stopping = False
        epoch = 0
        
        encoder = vgae_encoder(self.emb_size, model).cuda()
        decoder = vgae_decoder(self.emb_size, model).cuda()
        self.generator_1 = vgae(encoder, decoder).cuda()
        self.generator_2 = DenoisingNet(self.data.user_num, self.data.item_num, self.emb_size, self.model.getGCN(), self.model.getEmbeds()).cuda()
        self.generator_2.set_fea_adj(self.data.user_num+self.data.item_num, deepcopy(model.sparse_norm_adj).cuda())
        
        self.opt = torch.optim.Adam(model.parameters(), lr=self.lRate)
        self.opt_gen_1 = torch.optim.Adam(self.generator_1.parameters(), lr=self.lRate, weight_decay=0)
        self.opt_gen_2 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator_2.parameters()), lr=self.lRate, weight_decay=0, eps=1e-3)

        while not early_stopping:
            temperature = max(0.05, 2.0*pow(0.98,epoch))
            generate_loss_1, generate_loss_2, _bpr_loss, im_loss, ib_loss, reg_loss = 0, 0, 0, 0, 0, 0
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                data = deepcopy(model.sparse_norm_adj).cuda()
                user_idx, pos_idx, neg_idx = batch
                with torch.no_grad():
                    data1 = self.generator_1.generate(model.sparse_norm_adj)
                self.opt.zero_grad()
                self.opt_gen_1.zero_grad()
                self.opt_gen_2.zero_grad()
                    
                out1 = model.forward_graphcl(data1)
                out2 = model.forward_graphcl_(self.generator_2)
                
                loss = model.loss_graphcl(out1, out2, user_idx, pos_idx).mean() * self.ssl_reg
                im_loss += float(loss)
                loss.backward()
                
                self.opt.step()
                self.opt.zero_grad()
                
                # info bottleneck
                _out1 = model.forward_graphcl(data1)
                _out2 = model.forward_graphcl_(self.generator_2)
                
                loss_ib = model.loss_graphcl(_out1, out1.detach(), user_idx, pos_idx) + model.loss_graphcl(_out2, out2.detach(), user_idx, pos_idx)
                loss = loss_ib.mean() * self.ib_reg
                ib_loss += float(loss)
                loss.backward()
                
                self.opt.step()
                self.opt.zero_grad()
                
                # BPR
                usrEmbeds, itmEmbeds = model.forward_gcn(data)
                ancEmbeds = usrEmbeds[user_idx]
                posEmbeds = itmEmbeds[pos_idx]
                negEmbeds = itmEmbeds[neg_idx]
                bprLoss = bpr_loss(ancEmbeds, posEmbeds, negEmbeds)
                regLoss = l2_reg_loss(self.reg, model.embedding_dict['user_emb'], model.embedding_dict['item_emb'])
                loss = bprLoss + regLoss
                _bpr_loss += float(bprLoss)
                reg_loss += float(regLoss)
                loss.backward()
                
                loss_1 = self.generator_1(deepcopy(model.sparse_norm_adj).cuda(), user_idx, pos_idx, neg_idx)
                loss_2 = self.generator_2(user_idx, pos_idx, neg_idx, temperature)
                
                loss = loss_1 + loss_2
                generate_loss_1 += float(loss_1)
                generate_loss_2 += float(loss_2)
                loss.backward()
                
                self.opt.step()
                self.opt_gen_1.step()
                self.opt_gen_2.step()
                
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch:', n, 'generate_loss_1:', generate_loss_1, 'generate_loss_2:', generate_loss_2, \
                          'bpr_loss:', _bpr_loss, 'im_loss', im_loss, 'ib_loss', ib_loss)
            with torch.no_grad():
                self.user_emb, self.item_emb = model.forward_gcn(model.sparse_norm_adj)
            measure, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward_gcn(self.model.sparse_norm_adj)

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class AdaGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(AdaGCL_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.temp = 0.5 # default constant
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(self.layers)])
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward_gcn(self, adj):
        iniEmbeds = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], axis=0)
        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)
        
        return mainEmbeds[:self.data.user_num], mainEmbeds[self.data.user_num:]
    
    def forward_graphcl(self, adj):
        iniEmbeds = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], axis=0)
        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)
        
        return mainEmbeds
        
    def forward_graphcl_(self, generator):
        iniEmbeds = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], axis=0)
        embedsLst = [iniEmbeds]
        count = 0
        for gcn in self.gcnLayers:
            with torch.no_grad():
                adj = generator.generate(x=embedsLst[-1], layer=count)
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
            count += 1
        mainEmbeds = sum(embedsLst)
        
        return mainEmbeds
    
    def getEmbeds(self):
        for param in self.parameters():
            param.requires_grad = True
        return torch.concat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], axis=0)
    
    def loss_graphcl(self, x1, x2, users, items):
        T = self.temp
        user_embeddings1, item_embeddings1 = torch.split(x1, [self.data.user_num, self.data.item_num], dim=0)
        user_embeddings2, item_embeddings2 = torch.split(x2, [self.data.user_num, self.data.item_num], dim=0)

        user_embeddings1 = torch.nn.functional.normalize(user_embeddings1, dim=1)
        item_embeddings1 = torch.nn.functional.normalize(item_embeddings1, dim=1)
        user_embeddings2 = torch.nn.functional.normalize(user_embeddings2, dim=1)
        item_embeddings2 = torch.nn.functional.normalize(item_embeddings2, dim=1)

        user_embs1 = user_embeddings1[users]
        item_embs1 = item_embeddings1[items]
        user_embs2 = user_embeddings2[users]
        item_embs2 = item_embeddings2[items]

        all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)
        all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)

        all_embs1_abs = all_embs1.norm(dim=1)
        all_embs2_abs = all_embs2.norm(dim=1)
	
        sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs, all_embs2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)

        return loss
    def getEmbeds(self):
        self.unfreeze(self.gcnLayers)
        return torch.concat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], axis=0)

    def unfreeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = True

    def getGCN(self):
        return self.gcnLayers



