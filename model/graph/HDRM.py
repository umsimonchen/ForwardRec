import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import l2_reg_loss
import os
import numpy as np
import random
import pickle
import math
import enum
import sys
# paper: HDRM: Hyperbolic Diffusion Recommender Model. WWW'25
# https://github.com/yuanmeng-cpu/HDRM

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

def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    return Arcosh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def artanh(x):
    return Artanh.apply(x)

# Classes
class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5
    
class ManifoldParameter(nn.Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization.
    """
    def __new__(cls, data, requires_grad, manifold, c):
        return nn.Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c):
        self.c = c
        self.manifold = manifold

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(nn.Parameter, self).__repr__()
    
class OptimMixin(object):
    def __init__(self, *args, stabilize=None, **kwargs):
        self._stabilize = stabilize
        super().__init__(*args, **kwargs)

    def stabilize_group(self, group):
        pass

    def stabilize(self):
        """Stabilize parameters if they are off-manifold due to numerical reasons
        """
        for group in self.param_groups:
            self.stabilize_group(group)

class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """

    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2, c):
        """Squared distance between pairs of points."""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        raise NotImplementedError

    def proj(self, p, c):
        """Projects point p on the manifold."""
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p."""
        raise NotImplementedError

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, u, p, c):
        """Exponential map of u at point p."""
        raise NotImplementedError

    def logmap(self, p1, p2, c):
        """Logarithmic map of point p1 at point p2."""
        raise NotImplementedError

    def expmap0(self, u, c):
        """Exponential map of u at the origin."""
        raise NotImplementedError

    def logmap0(self, p, c):
        """Logarithmic map of point p at the origin."""
        raise NotImplementedError

    def mobius_add(self, x, y, c, dim=-1):
        """Adds points x and y."""
        raise NotImplementedError

    def mobius_matvec(self, m, x, c):
        """Performs hyperboic martrix-vector multiplication."""
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weigths on the manifold."""
        raise NotImplementedError

    def inner(self, p, c, u, v=None, keepdim=False):
        """Inner product for tangent vectors at point x."""
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        """Parallel transport of u from x to y."""
        raise NotImplementedError

    def ptransp0(self, x, u, c):
        """Parallel transport of u from the origin to y."""
        raise NotImplementedError


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.

    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

    c = 1 / K is the hyperbolic curvature.
    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))
    
    def fermi_dirac_decoder(self,x,t,r):
        # 计算分母: exp(d^2 - r)/t + 1
        denominator = torch.exp(x - r) / t + 1
        # 计算最终评分: 1 / denominator
        scores = 1.0 / denominator
        return scores

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        return torch.clamp(sqdist, max=50.0)

    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def proj_tan(self, u, x, c):
        d = x.size(1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
        return vals + mask * u

    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)

    def logmap(self, x, y, c):
        K = 1. / c
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def egrad2rgrad(self, x, grad, k, dim=-1):
        grad.narrow(-1, 0, 1).mul_(-1)
        grad = grad.addcmul(self.inner(x, grad, dim=dim, keepdim=True), x / k)
        return grad

    def inner(self, u, v, keepdim: bool = False, dim: int = -1):
        d = u.size(dim) - 1
        uv = u * v
        if keepdim is False:
            return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(
                dim, 1, d
            ).sum(dim=dim, keepdim=False)
        else:
            return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
                dim=dim, keepdim=True
            )

class RiemannianSGD(OptimMixin, torch.optim.SGD):
    r"""
    Riemannian Stochastic Gradient Descent with the same API as :class:`torch.optim.SGD`.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float
        learning rate
    momentum : float (optional)
        momentum factor (default: 0)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    dampening : float (optional)
        dampening for momentum (default: 0)
    nesterov : bool (optional)
        enables Nesterov momentum (default: False)

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)
    """
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]
                learning_rate = group["lr"]

                for point in group["params"]:

                    if isinstance(point, ManifoldParameter):
                        manifold = point.manifold
                        c = point.c
                    else:
                        manifold = Hyperboloid()
                        c = 1.

                    grad = point.grad

                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianSGD does not support sparse gradients, use SparseRiemannianSGD instead"
                        )
                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        if momentum > 0:
                            state["momentum_buffer"] = grad.clone()

                    grad.add_(point, alpha=weight_decay)
                    grad = manifold.egrad2rgrad(point, grad, c)

                    if momentum > 0:
                        momentum_buffer = state["momentum_buffer"]
                        momentum_buffer.mul_(momentum).add_(grad, alpha=1 - dampening)
                        if nesterov:
                            grad = grad.add_(momentum, momentum_buffer)
                        else:
                            grad = momentum_buffer

                        new_point = manifold.expmap(-learning_rate * grad, point, c)

                        components = new_point[:, 1:]
                        dim0 = torch.sqrt(torch.sum(components * components, dim=1, keepdim=True) + 1)
                        new_point = torch.cat([dim0, components], dim=1)

                        new_momentum_buffer = manifold.ptransp(point, new_point, momentum_buffer, c)
                        momentum_buffer.set_(new_momentum_buffer)

                        # use copy only for user facing point
                        copy_or_set_(point, new_point)
                    else:
                        # new_point = manifold.retr(point, -learning_rate * grad)
                        new_point = manifold.expmap(-learning_rate * grad, point, c)

                        components = new_point[:, 1:]
                        dim0 = torch.sqrt(torch.sum(components * components, dim=1, keepdim=True) + 1)
                        new_point = torch.cat([dim0, components], dim=1)

                        copy_or_set_(point, new_point)

                    group["step"] += 1
                if self._stabilize is not None and group["step"] % self._stabilize == 0:
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, ManifoldParameter):
                continue
            manifold = p.manifold
            momentum = group["momentum"]
            copy_or_set_(p, manifold.proj(p))
            if momentum > 0:
                param_state = self.state[p]
                if not param_state:  # due to None grads
                    continue
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                    buf.set_(manifold.proju(p, buf))

class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, act, time_type="cat", norm=False, dropout=0.5):
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

    def forward(self, noise_emb, con_emb, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(noise_emb.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            noise_emb = nn.functional.normalize(noise_emb)
        noise_emb = self.drop(noise_emb)

        all_emb = torch.cat([noise_emb, emb, con_emb], dim=-1)

        for i, layer in enumerate(self.in_layers):
            all_emb = layer(all_emb)
            if self.act == 'tanh':
                all_emb = torch.tanh(all_emb)
            elif self.act == 'sigmoid':
                all_emb = torch.sigmoid(all_emb)
            elif self.act == 'relu':
                all_emb = nn.functional.relu(all_emb)
        for i, layer in enumerate(self.out_layers):
            all_emb = layer(all_emb)
            if i != len(self.out_layers) - 1:
                if self.act == 'tanh':
                    all_emb = torch.tanh(all_emb)
                elif self.act == 'sigmoid':
                    all_emb = torch.sigmoid(all_emb)
                elif self.act == 'relu':
                    all_emb = nn.functional.relu(all_emb)
        return all_emb

class GaussianDiffusion(nn.Module):
    def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max, \
                 steps, device, restrict=False, history_num_per_term=10, beta_fixed=True):
        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.restrict = restrict
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
            # assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"

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
                return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
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
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(
            self.device)  # alpha_{t-1}
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(
            self.device)  # alpha_{t+1}
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

        mse = mean_flat((target - model_output) ** 2)

        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = torch.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / (
                            (1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts]))
                weight = torch.where((ts == 0), 1.0, weight)
                likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output)) ** 2 / 2.0)
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
            pt_all *= 1 - uniform_prob
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

    def get_statistics(self, x_start):
        mu = torch.mean(x_start, dim=0)
        sigma = torch.std(x_start, dim=0)
        return mu, sigma

    def transform_noise(self, noise, x_start):
        # (5): ε' = sgn(x₀) ⊙ |ε|
        epsilon_prime = torch.sign(x_start) * torch.abs(noise)
        # (6): ε̄ = μ + σ ⊙ ε
        mu, sigma = self.get_statistics(x_start)
        epsilon_bar = mu + sigma * epsilon_prime
        return epsilon_bar

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        if self.restrict:
            directional_noise = self.transform_noise(noise, x_start)
            return (
                    self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                    + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                    * directional_noise
            )
        else:
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
        assert t.shape == (B,)
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
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        if timesteps[0] >= len(arr):
            res = arr[-1].float()
        else:
            res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def get_reconstruct_loss(self, cat_emb, re_emb, pt):
        loss = mean_flat((cat_emb - re_emb) ** 2)
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

        # loss = loss.mean()
        return loss

# Functions
def copy_or_set_(dest, source):
    """
    A workaround to respect strides of :code:`dest` when copying :code:`source`
    (https://github.com/geoopt/geoopt/issues/70)
    Parameters
    ----------
    dest : torch.Tensor
        Destination tensor where to store new data
    source : torch.Tensor
        Source data to put in the new tensor
    Returns
    -------
    dest
        torch.Tensor, modified inplace
    """
    if dest.stride() != source.stride():
        return dest.copy_(source)
    else:
        return dest.set_(source)
    
def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
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

# def normal_kl(mean1, logvar1, mean2, logvar2):
#     """
#     Compute the KL divergence between two gaussians.

#     Shapes are automatically broadcasted, so batches can be compared to
#     scalars, among other use cases.
#     """
#     tensor = None
#     for obj in (mean1, logvar1, mean2, logvar2):
#         if isinstance(obj, torch.Tensor):
#             tensor = obj
#             break
#     assert tensor is not None, "at least one argument must be a Tensor"

#     # Force variances to be Tensors. Broadcasting helps convert scalars to
#     # Tensors, but it does not work for th.exp().
#     logvar1, logvar2 = [
#         x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
#         for x in (logvar1, logvar2)
#     ]

#     return 0.5 * (
#             -1.0
#             + logvar2
#             - logvar1
#             + torch.exp(logvar1 - logvar2)
#             + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
#     )

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=1)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class HDRM(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(HDRM, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['HDRM'])
        self.n_layers = int(args['-n_layer'])
        self.c = int(args['-c']) # curvature, [-1,1]
        self.steps = int(args['-steps']) # diffusion steps, [10,20,30,40,50,60]
        self.noise_scale = float(args['-noise_scale']) # diffusion noise scale, [0.0001, 0.01]
        self.noise_min = float(args['-noise_min']) # diffusion noise min, [1e-4, 1e-3]
        self.noise_max = float(args['-noise_max']) # diffusion noise max, [1e-3, 1e-2]
        self.alpha = float(args['-alpha']) # loss balance factor, [0.1,0.2,0.3,0.4,0.5,0.6]
        self.gamma = float(args['-gamma']) # denoising reweight, [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        self.margin = float(args['-margin']) # hyperbolic margin, [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
        self.pretrain = True # default constant
        if self.pretrain:
            dataset = self.config['training.set'].split('/')[2]
        
        out_dims = eval('[200,1000]') + [self.emb_size] # default constant, but [[300], [200,600], [1000]]
        in_dims = out_dims[::-1]
        time_emb_size = 10 # default constant
        self.act = 'relu' # default constant, but in ['tanh','sigmoid','relu']
        self.user_reverse_model = DNN(in_dims, out_dims, time_emb_size, act=self.act).cuda()
        self.item_reverse_model = DNN(in_dims, out_dims, time_emb_size, act=self.act).cuda()
        
        mean_type = "x0" # ["x0", "eps"] # default constant, but ['x0', eps]
        if mean_type == 'x0':
            self.mean_type = ModelMeanType.START_X
        elif mean_type == 'eps':
            self.mean_type = ModelMeanType.EPSILON
        else:
            raise ValueError("Unimplemented mean type %s" % args.mean_type)
        
        self.noise_schedule = "linear-var" # default constant, but ["linear", "linear-var", "cosine", "binomial"]
        self.device = torch.device('cuda')
        self.diffusion = GaussianDiffusion(self.mean_type, self.noise_schedule, self.noise_scale, self.noise_min, self.noise_max, self.steps, self.device)
        self.model = HDRM_Encoder(self.data, self.emb_size, self.n_layers, self.steps, dataset, self.c, self.margin)
        
    def train(self):
        model = self.model.cuda()
        opt = RiemannianSGD(model.parameters(), lr=1e-5, weight_decay=0.01, momentum=0.95)
        opt_user_dnn = torch.optim.Adam(self.user_reverse_model.parameters(), lr=self.lRate)
        opt_item_dnn = torch.optim.Adam(self.item_reverse_model.parameters(), lr=self.lRate)
        early_stopping = False
        epoch = 0
        # pretrain
        for param in model.parameters():
            param.requires_grad = False
        while not early_stopping:
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                # Batch Normalization & + Dropout
                model.train()
                self.user_reverse_model.train()
                self.item_reverse_model.train()
                
                user_idx, pos_idx, neg_idx = batch
                loss, re_loss, pos_score = model.bpr_loss(user_idx, pos_idx, neg_idx, self.user_reverse_model, self.item_reverse_model, self.diffusion)
                
                re_loss = torch.sum(re_loss)
                loss = (1-self.alpha) * loss + self.alpha * re_loss
                # pos_score_ = torch.sigmoid(pos_score).detach()
                # weight = torch.pow(pos_score_, self.gamma)
                # loss = loss * weight
                # loss = loss.mean()
                 
                # Backward and optimize
                opt_user_dnn.zero_grad()
                opt_item_dnn.zero_grad()
                opt.zero_grad()
                loss.backward()
                opt_item_dnn.step()
                opt_user_dnn.step()
                opt.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'loss:', loss.item())
            with torch.no_grad():
                # Batch Normalization & + Dropout
                model.eval()
                self.user_reverse_model.eval()
                self.item_reverse_model.eval()
                self.valid_rating = model.getUsersRating(list(range(self.data.user_num)), self.user_reverse_model, self.item_reverse_model, self.diffusion)
            measure, early_stopping = self.fast_evaluation(epoch)
            epoch += 1
        self.valid_rating = self.best_valid_rating
        with open('performance.txt','a') as fp:
            fp.write(str(self.bestPerformance[1])+"\n")
        
    def save(self):
        with torch.no_grad():
            self.best_valid_rating = self.valid_rating

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = self.valid_rating[u]
        return score.cpu().numpy()

class HDRM_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, steps, dataset, c, margin):
        super(HDRM_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.n_layers = n_layers
        self.steps = steps
        self.dataset = dataset
        self.c = c
        self.margin = margin
        self.manifold = getattr(sys.modules[__name__], "Hyperboloid")()
        self.sampling_steps = int(1*self.steps) # default constant, but [0, T/4, T/2]
        self._init_weight()
        
    def _init_weight(self):
        if self.dataset:
            with open('pretrain/emb_HGCF_%s'%self.dataset, 'rb') as fp:
                self.embedding = pickle.load(fp)
            self.embedding_user = nn.Embedding.from_pretrained(torch.tensor(self.embedding[:self.data.user_num]))
            self.embedding_item = nn.Embedding.from_pretrained(torch.tensor(self.embedding[self.data.user_num:]))
        else:
            initializer = nn.init.xavier_uniform_
            self.embedding_user = nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size)))
            self.embedding_item = nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size)))
            
            print('use NORMAL distribution initilizer')
            self.embedding_user.clone().uniform_(-0.1, 0.1)
            self.embedding_user = nn.Parameter(self.manifold.expmap0(self.embedding_user, self.c))
            self.embedding_user = ManifoldParameter(self.embedding_user, True, self.manifold, self.c)
    
            self.embedding_item.clone().uniform_(-0.1, 0.1)
            self.embedding_item = nn.Parameter(self.manifold.expmap0(self.embedding_item, self.c))
            self.embedding_item = ManifoldParameter(self.embedding_item, True, self.manifold, self.c)
        
        self.f = nn.Sigmoid()
        self.Graph = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.norm_adj).cuda()
        self.avg_Graph = TorchGraphInterface.convert_sparse_mat_to_tensor(self.data.interaction_mat.multiply(1./self.data.interaction_mat.sum(axis=1))).cuda()
        
    def computer(self,diff_model, user_reverse_model, item_reverse_model,  user, pos):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        users_emb_hpy = self.manifold.proj(users_emb, c=self.c)
        items_emb_hpy = self.manifold.proj(items_emb, c=self.c)

        users_emb = self.manifold.logmap0(users_emb_hpy, c=self.c)
        items_emb = self.manifold.logmap0(items_emb_hpy, c=self.c)


        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        
        g_droped = self.Graph    
        
        
        for i in range(self.n_layers):
            # print(all_emb.shape)
            embs.append(torch.spmm(g_droped, embs[i]))

        light_out = sum(embs[1:])
        
        users, items = torch.split(light_out, [self.data.user_num, self.data.item_num])

        # get batch user and item emb
        ori_user_emb = users[user]
        ori_item_emb = items[pos]
        
        # add noise to user and item
        noise_user_emb, noise_item_emb, ts, pt = self.apply_noise(ori_user_emb, ori_item_emb, diff_model)
        # reverse
        user_model_output = user_reverse_model(noise_user_emb, ori_item_emb, ts)
        item_model_output = item_reverse_model(noise_item_emb, ori_user_emb, ts)
        
        epsilon = 0.05
        user_model_output = (1-epsilon) * ori_user_emb + epsilon * user_model_output
        item_model_output = (1-epsilon) * ori_item_emb + epsilon * item_model_output


        # get recons loss = mse, similar to diffusion, while pt is useless
        user_recons = diff_model.get_reconstruct_loss(ori_user_emb, user_model_output, pt)
        item_recons = diff_model.get_reconstruct_loss(ori_item_emb, item_model_output, pt)
        recons_loss = (user_recons + item_recons) / 2

        # update the batch user and item emb
        return user_model_output, item_model_output,recons_loss, items
    
    def computer_infer(self, user, diff_model, user_reverse_model, item_reverse_model, sampling_noise=False):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        users_emb_hpy = self.manifold.proj(users_emb, c=self.c)
        items_emb_hpy = self.manifold.proj(items_emb, c=self.c)

        users_emb = self.manifold.logmap0(users_emb_hpy, c=self.c)
        items_emb = self.manifold.logmap0(items_emb_hpy, c=self.c)

        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        g_droped = self.Graph  
          
        for i in range(self.n_layers):
            # print(all_emb.shape)
            embs.append(torch.spmm(g_droped, embs[i]))

        light_out = sum(embs[1:])
        users, items = torch.split(light_out, [self.data.user_num, self.data.item_num])

        user_emb = users[user]  
        all_aver_item_emb = torch.sparse.mm(self.avg_Graph, items) # Efficient mean-pooling
        # all_aver_item_emb = []
        # for pos_item in allPos:
        #     item_emb = items[pos_item]
        #     aver_item_emb = torch.mean(item_emb, dim=0)
        #     all_aver_item_emb.append(aver_item_emb)
        # all_aver_item_emb = torch.stack(all_aver_item_emb).to(users.device)

        noise_user_emb = user_emb

        # get generated item          
        # reverse
        noise_emb = self.apply_T_noise(all_aver_item_emb, diff_model)
        indices = list(range(self.sampling_steps))[::-1]
        for i in indices:
            t = torch.tensor([i] * noise_emb.shape[0]).to(noise_emb.device)
            out = diff_model.p_mean_variance(item_reverse_model, noise_emb, noise_user_emb, t)
            if sampling_noise:
                noise = torch.randn_like(noise_emb)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(noise_emb.shape) - 1)))
                )  # no noise when t == 0
                noise_emb = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                noise_emb = out["mean"]
                
        epsilon = 0.05
        noise_emb = (1-epsilon) * user_emb + epsilon * noise_emb

        return noise_emb, items
    
    def getUsersRating(self, users, user_reverse_model, item_reverse_model, diff_model):
        item_emb, all_items = self.computer_infer(users, diff_model, user_reverse_model, item_reverse_model)

        item_emb = self.manifold.proj(self.manifold.expmap0(item_emb, c=self.c), c=self.c)
        all_items = self.manifold.proj(self.manifold.expmap0(all_items, c=self.c), c=self.c)

        rating = self.rounding_inner(item_emb, all_items)
        return rating
    
    def rounding_inner(self, item_emb, all_items):
        num_users=item_emb.shape[0]
        num_items=all_items.shape[0]

        probs_matrix = np.zeros((num_users, num_items))
        for i in range(num_users):
            emb_in = item_emb[i, :]
            emb_in = emb_in.repeat(num_items).view(num_items, -1)

            emb_out = all_items
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
            probs = sqdist.detach().cpu().numpy() * -1
            probs_matrix[i] = np.reshape(probs, [-1, ])

        return torch.from_numpy(probs_matrix)
    
    def getEmbedding(self, users, pos_items, neg_items, user_reverse_model, item_reverse_model, diff_model):
        users_emb, pos_emb,recons_loss, all_items = self.computer(diff_model, user_reverse_model, item_reverse_model, users, pos_items)

        users_emb = self.manifold.proj(self.manifold.expmap0(users_emb, c=self.c), c=self.c)
        pos_emb = self.manifold.proj(self.manifold.expmap0(pos_emb, c=self.c), c=self.c)
        all_items = self.manifold.proj(self.manifold.expmap0(all_items, c=self.c), c=self.c)

        neg_emb = all_items[neg_items]

        return users_emb, pos_emb, neg_emb,recons_loss
    
    def bpr_loss(self, users, pos, neg, user_reverse_model, item_reverse_model, diff_model):
        users_emb, pos_emb, neg_emb,reconstruct_loss = self.getEmbedding(users, pos, neg, user_reverse_model, item_reverse_model, diff_model)



        pos_scores = self.manifold.sqdist(users_emb, pos_emb, self.c)
        neg_scores = self.manifold.sqdist(users_emb, neg_emb, self.c)
        
        loss = pos_scores - neg_scores + self.margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss,reconstruct_loss,pos_scores
    
    def apply_T_noise(self, cat_emb, diff_model):
        t = torch.tensor([self.steps - 1] * cat_emb.shape[0]).to(cat_emb.device)
        noise = torch.randn_like(cat_emb)
        noise_emb = diff_model.q_sample(cat_emb, t, noise)
        return noise_emb
    
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
        