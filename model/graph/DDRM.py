import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from data.augmentor import GraphAugmentor
import os
import numpy as np
import random
import pickle
import math
import enum

# Paper: Denoising Diffusion Recommender Model. SIGIR'24
# https://github.com/Polaris-JZ/DDRM

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

class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5, act='tanh'):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.act = act

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0]*2 + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    # def forward(self, x, timesteps):
    #     time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
    #     emb = self.emb_layer(time_emb)
    #     if self.norm:
    #         x = F.normalize(x)
    #     x = self.drop(x)

    #     h = torch.cat([x, emb], dim=-1)

    #     for i, layer in enumerate(self.in_layers):
    #         h = layer(h)
    #         h = torch.tanh(h)
    #     for i, layer in enumerate(self.out_layers):
    #         h = layer(h)
    #         if i != len(self.out_layers) - 1:
    #             h = torch.tanh(h)
    #     return h

    def forward(self, noise_emb, con_emb, timesteps):
        self.act = 'tanh'
        time_emb = self.timestep_embedding(timesteps, self.time_emb_dim).to(noise_emb.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            noise_emb = F.normalize(noise_emb)
        noise_emb = self.drop(noise_emb)

        all_emb = torch.cat([noise_emb, emb, con_emb], dim=-1)

        for i, layer in enumerate(self.in_layers):
            all_emb = layer(all_emb)
            if self.act == 'tanh':
                all_emb = torch.tanh(all_emb)
            elif self.act == 'sigmoid':
                all_emb = torch.sigmoid(all_emb)
            elif self.act == 'relu':
                all_emb = F.relu(all_emb)
        for i, layer in enumerate(self.out_layers):
            all_emb = layer(all_emb)
            if i != len(self.out_layers) - 1:
                if self.act == 'tanh':
                    all_emb = torch.tanh(all_emb)
                elif self.act == 'sigmoid':
                    all_emb = torch.sigmoid(all_emb)
                elif self.act == 'relu':
                    all_emb = F.relu(all_emb)
        return all_emb

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
    
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
    
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class GaussianDiffusion(nn.Module):
    def __init__(self, config, mean_type, noise_schedule, noise_scale, noise_min, noise_max,\
            steps, device, history_num_per_term=10, beta_fixed=True):
        self.config = config
        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device

        self.history_num_per_term = history_num_per_term
        self.Lt_history = torch.zeros(steps, history_num_per_term, dtype=torch.float64).to(device)
        self.Lt_count = torch.zeros(steps, dtype=int).to(device)

        if noise_scale != 0.:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(self.device)
            if beta_fixed:
                self.betas[0] = 0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
                # The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"

            self.calculate_for_diffusion()

        super(GaussianDiffusion, self).__init__()
    
    def get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return self.betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            return self.betas_for_alpha_bar(
            self.steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        elif self.noise_schedule == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")
    
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  # alpha_{t-1}
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(self.device)  # alpha_{t+1}
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    def p_sample(self, model, x_start, steps, sampling_noise=False):
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.:
            for i in indices:
                t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t)
            return x_t

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t
    
    def training_losses(self, model, x_start, reweight=False):
        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, 'importance')
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        terms = {}
        model_output = model(x_t, ts)
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = self.mean_flat((target - model_output) ** 2)

        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = torch.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
                weight = torch.where((ts == 0), 1.0, weight)
                likelihood = self.mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
                loss = torch.where((ts == 0), likelihood, mse)
        else:
            weight = torch.tensor([1.0] * len(target)).to(device)

        terms["loss"] = weight * loss
        
        # update Lt_history & Lt_count
        for t, loss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss)
                    raise ValueError

        terms["loss"] /= pt
        return terms

    def sample_timesteps(self, batch_size, method='uniform', uniform_prob=0.001):
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, method='uniform')
            
            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1- uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt
        
        elif method == 'uniform':  # uniform sampling
            t = torch.randint(0, self.steps, (batch_size,), device=self.device).long()
            pt = torch.ones_like(t).float()
            return t, pt
        else:
            raise ValueError
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x, con_emb, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B, )
        model_output = model(x, con_emb, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        
        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        if timesteps[0] >= len(arr):
            res = arr[-1].float()
        else:
            res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def get_reconstruct_loss(self, cat_emb, re_emb, pt):
        loss = self.mean_flat((cat_emb - re_emb) ** 2)
        # print(loss.shape)
        # print(pt)
        # loss /= pt

        # # update Lt_history & Lt_count
        # for t, loss in zip(ts, loss):
        #     if self.Lt_count[t] == self.history_num_per_term:
        #         Lt_history_old = self.Lt_history.clone()
        #         self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
        #         self.Lt_history[t, -1] = loss.detach()
        #     else:
        #         try:
        #             self.Lt_history[t, self.Lt_count[t]] = loss.detach()
        #             self.Lt_count[t] += 1
        #         except:
        #             print(t)
        #             print(self.Lt_count[t])
        #             print(loss)
        #             raise ValueError

        loss = loss.mean()
        return loss
    
    def betas_from_linear_variance(self, steps, variance, max_beta=0.999):
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
        return np.array(betas)

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].
    
        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                          produces the cumulative product of (1-beta) up to that
                          part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                         prevent singularities.
        """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)
    
    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two gaussians.
    
        Shapes are automatically broadcasted, so batches can be compared to
        scalars, among other use cases.
        """
        tensor = None
        for obj in (mean1, logvar1, mean2, logvar2):
            if isinstance(obj, torch.Tensor):
                tensor = obj
                break
        assert tensor is not None, "at least one argument must be a Tensor"
    
        # Force variances to be Tensors. Broadcasting helps convert scalars to
        # Tensors, but it does not work for torch.exp().
        logvar1, logvar2 = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
            for x in (logvar1, logvar2)
        ]
    
        return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )
    
    def mean_flat(self, tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=1)

class DDRM(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(DDRM, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['DDRM'])
        drop_rate = float(args['-droprate'])
        n_layers = int(args['-n_layer'])
        
        # diffusion default setting
        out_dims = eval('[200,600]') + [self.emb_size]
        in_dims = out_dims[::-1]
        
        user_reverse_model = DNN(in_dims, out_dims, self.emb_size, time_type='cat').cuda()
        item_reverse_model = DNN(in_dims, out_dims, self.emb_size, time_type='cat').cuda()
        
        self.mean_type = "x0" # ["x0", "eps"]
        self.noise_schedule = "linear-var" # ["linear", "linear-var", "cosine", "binomial"]
        self.noise_scale = 5e-4 # [0, 1e-5, 1e-4, 5e-3, 1e-2, 1e-1]
        self.noise_min = 0.005 # [5e-4, 1e-3, 5e-3]
        self.noise_max = 0.01 # [5e-3, 1e-2]
        self.steps = 2 # [2,5,10,40,50,100]
        self.aplha = 0.1 # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.device = torch.device('cuda')
        if self.mean_type == "x0":
            mean_type = ModelMeanType.START_X
        elif self.mean_type == "eps":
            mean_type = ModelMeanType.EPSILON
        else:
            raise ValueError("Unimplemented mean type %s"%mean_type)
            
        diffusion = GaussianDiffusion(args, mean_type, self.noise_schedule, self.noise_scale, self.noise_min, self.noise_max, self.steps, self.device).cuda()
        self.model = DDRM_Encoder(self.data, self.emb_size, drop_rate, n_layers, user_reverse_model, item_reverse_model, diffusion)
        
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        while not early_stopping:
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                user_emb0, pos_item_emb0, neg_item_emb0 = model.embedding_dict['user_emb'][user_idx], model.embedding_dict['item_emb'][pos_idx], model.embedding_dict['item_emb'][neg_idx]  
                user_model_output, item_model_output, recons_loss = model.cal_loss(user_emb, pos_item_emb)
                rec_loss = bpr_loss(user_model_output, item_model_output, neg_item_emb)
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb0, pos_item_emb0, neg_item_emb0) + recons_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())
            with torch.no_grad():
                ori_user_emb, self.item_emb = self.model()
                self.user_emb = self.model.inference(ori_user_emb, self.item_emb)

            measure, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")    

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()
    
class DDRM_Encoder(nn.Module):
    def __init__(self, data, emb_size, drop_rate, n_layers, user_reverse_model, item_reverse_model, diffusion):
        super(DDRM_Encoder, self).__init__()
        self.data = data
        self.drop_rate = drop_rate
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.norm_inter = data.norm_inter
        self.user_reverse_model = user_reverse_model
        self.item_reverse_model = item_reverse_model
        self.diff_model = diffusion
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.sparse_norm_inter = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_inter).cuda()
        self.steps = 2 # default constant
        self.sampling_steps = 2 # default constant
        self.sampling_noise = False # default constant

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def graph_reconstruction(self, aug_type=1):
        if aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self, aug_type=1):
        dropped_mat = None
        if aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.data.interaction_mat, self.drop_rate)
        elif aug_type == 1 or aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.drop_rate)
        dropped_mat = self.data.convert_to_laplacian_mat(dropped_mat)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        
        if self.drop_rate > 0:
            g_dropped = self.graph_reconstruction()
        else:
            g_dropped = self.sparse_norm_adj
            
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(g_dropped, ego_embeddings)
            ego_embeddings = torch.nn.functional.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings
    
    def cal_loss(self, ori_user_emb, ori_item_emb):
        # add noise to user and item
        noise_user_emb, noise_item_emb, ts, pt = self.apply_noise(ori_user_emb, ori_item_emb, self.diff_model)        
        
        # reverse
        user_model_output = self.user_reverse_model(noise_user_emb, ori_item_emb, ts)
        item_model_output = self.item_reverse_model(noise_item_emb, ori_user_emb, ts)
        
        # get recons loss
        user_recons = self.diff_model.get_reconstruct_loss(ori_user_emb, user_model_output, pt)
        item_recons = self.diff_model.get_reconstruct_loss(ori_item_emb, item_model_output, pt)
        recons_loss = (user_recons + item_recons) / 2

        return user_model_output, item_model_output, recons_loss       

    def apply_noise(self, user_emb, item_emb, diff_model):
        # cat_emb shape: (batch_size*3, emb_size)
        emb_size = user_emb.shape[0]
        ts, pt = diff_model.sample_timesteps(emb_size, 'uniform')
        # ts_ = torch.tensor([self.config['steps'] - 1] * cat_emb.shape[0]).to(cat_emb.device)

        # add noise to users
        user_noise = torch.randn_like(user_emb)
        item_noise = torch.randn_like(item_emb)
        user_noise_emb = diff_model.q_sample(user_emb, ts, user_noise)
        item_noise_emb = diff_model.q_sample(item_emb, ts, item_noise)
        return user_noise_emb, item_noise_emb, ts, pt
    
    def apply_T_noise(self, cat_emb, diff_model):
        t = torch.tensor([self.config['steps'] - 1] * cat_emb.shape[0]).cuda()
        noise = torch.randn_like(cat_emb)
        noise_emb = diff_model.q_sample(cat_emb, t, noise)
        return noise_emb
    
    def inference(self, ori_user_emb, ori_item_emb):
        all_aver_item_emb = torch.sparse.mm(self.sparse_norm_inter, ori_item_emb)
        noise_user_emb = ori_user_emb
        
        # get generated item
        # reverse
        noise_emb = self.apply_T_noise(all_aver_item_emb, self.diff_model)
        indices = list(range(self.sampling_steps))[::-1]
        for i in indices:
            t = torch.tensor([i] * noise_emb.shape[0]).cuda()
            out = self.diff_model.p_mean_variance(self.item_reverse_model, noise_emb, noise_user_emb, t)
            if self.sampling_noise:
                noise = torch.randn_like(noise_emb)
                nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_emb.shape) - 1))))  # no noise when t == 0
                noise_emb = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                noise_emb = out["mean"]
        return noise_emb
        
        
        
        
        
        
        
        
        