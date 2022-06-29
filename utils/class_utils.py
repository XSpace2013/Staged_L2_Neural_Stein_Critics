'''
Pulled from "Learning the Stein Discrepancy for Training and 
Evaluating Energy-Based Models without Sampling" from
Grathwohl et al.
'''
import torch
import torch.nn as nn
import torch.distributions as distributions
import numpy as np


class GaussianMixture(nn.Module):
	def __init__(self, mix, comp, dim=1):
		super(GaussianMixture, self).__init__()
		self.dist = distributions.MixtureSameFamily(mix, comp)
		self.dim = dim

	def sample(self, n):
		return self.dist.sample((n,)).view(-1,self.dim)

	def forward(self, x):
		return self.dist.log_prob(x).view(x.size(0), -1).sum(1)


class EBM(nn.Module):
    def __init__(self, net, base_dist=None, learn_base_dist=True):
        super(EBM, self).__init__()
        self.net = net
        if base_dist is not None:
            self.base_mu = nn.Parameter(base_dist.loc, requires_grad=learn_base_dist)
            self.base_logstd = nn.Parameter(base_dist.scale.log(), requires_grad=learn_base_dist)
            self.base_logweight = nn.Parameter(base_dist.scale.mean() * 0., requires_grad=learn_base_dist)
        else:
            self.base_mu = None
            self.base_logstd = None

    def forward(self, x, lp=False):
        if self.base_mu is None:
            bd = 0
        else:
            base_dist = distributions.Normal(self.base_mu, self.base_logstd.exp())
            bd = base_dist.log_prob(x).view(x.size(0), -1).sum(1)
        net = self.net(x)
        if lp:
            return net + bd, net
        else:
            return net + bd

    def sample(self, x_init, l=1., e=.01, n_steps=100, anneal=None):
        x_k = torch.autograd.Variable(x_init, requires_grad=True)
        # sgld
        if anneal == "lin":
            lrs = list(reversed(np.linspace(e, l, n_steps)))
        elif anneal == "log":
            lrs = np.logspace(np.log10(l), np.log10(e))
        else:
            lrs = [l for _ in range(n_steps)]
        for this_lr in lrs:
            f_prime = torch.autograd.grad(self(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += this_lr * f_prime + torch.randn_like(x_k) * e
        final_samples = x_k.detach()
        return final_samples


def randb(size):
    dist = distributions.Bernoulli(probs=(.5 * torch.ones(*size)))
    return dist.sample().float()


class GaussianBernoulliRBM(nn.Module):
    def __init__(self, B, b, c, burn_in=2000):
        super(GaussianBernoulliRBM, self).__init__()
        self.B = nn.Parameter(B)
        self.b = nn.Parameter(b)
        self.c = nn.Parameter(c)
        self.dim_x = B.size(0)
        self.dim_h = B.size(1)
        self.burn_in = burn_in

    def score_function(self, x):  # dlogp(x)/dx
        return .5 * torch.tanh(.5 * x @ self.B + self.c) @ self.B.t() + self.b - x

    def forward(self, x):  # logp(x)
        B = self.B
        b = self.b
        c = self.c
        xBc = (0.5 * x @ B) + c
        unden =  (x * b).sum(1) - .5 * (x ** 2).sum(1)# + (xBc.exp() + (-xBc).exp()).log().sum(1)
        unden2 = (x * b).sum(1) - .5 * (x ** 2).sum(1) + torch.tanh(xBc/2.).sum(1)#(xBc.exp() + (-xBc).exp()).log().sum(1)
        print((unden - unden2).mean())
        assert len(unden) == x.shape[0]
        return unden

    def sample(self, n):
        x = torch.randn((n, self.dim_x)).to(self.B)
        h = (randb((n, self.dim_h)) * 2. - 1.).to(self.B)
        for t in range(self.burn_in):
            x, h = self._blocked_gibbs_next(x, h)
        x, h = self._blocked_gibbs_next(x, h)
        return x

    def _blocked_gibbs_next(self, x, h):
        """
        Sample from the mutual conditional distributions.
        """
        B = self.B
        b = self.b
        # Draw h.
        XB2C = (x @ self.B) + 2.0 * self.c
        # Ph: n x dh matrix
        Ph = torch.sigmoid(XB2C)
        # h: n x dh
        h = (torch.rand_like(h) <= Ph).float() * 2. - 1.
        assert (h.abs() - 1 <= 1e-6).all().item()
        # Draw X.
        # mean: n x dx
        mean = h @ B.t() / 2. + b
        x = torch.randn_like(mean) + mean
        return x, h


class LambdaStager():
    def __init__(self, init_lam, beta, min_lam):

        self.lams = [init_lam]
        self.beta = beta
        self.min_lam = min_lam

    def get_lam(self):
        return self.lams[-1]

    def update(self):
        self.lams.append(max(self.min_lam, self.lams[-1] * self.beta))


class Swish(nn.Module):
    def __init__(self, dim=-1):
        super(Swish, self).__init__()
        if dim > 0:
            self.beta = nn.Parameter(torch.ones((dim,)))
        else:
            self.beta = torch.ones((1,))

    def forward(self, x):
        if len(x.size()) == 2:
            return x * torch.sigmoid(self.beta[None, :] * x)
        else:
            return x * torch.sigmoid(self.beta[None, :, None, None] * x)


def init_weights(module):
    if isinstance(module, nn.Linear):
        module.weight.data = module.weight.data
        module.bias.data = module.bias.data*0


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, nn_layers, activation='swish', beta=5, dropout=True, init=False):
        assert nn_layers >= 1
        assert activation in ['swish','softplus']
        
        super(MLP, self).__init__()

        layers = []
        
        layers.append(nn.Linear(in_dim, hidden_dim))
        if activation == 'swish':
            layers.append(Swish(hidden_dim))
        elif activation == 'softplus':
            layers.append(nn.Softplus(hidden_dim, beta))
        if dropout:
            layers.append(nn.Dropout(0.5))
        
        for _ in range(nn_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if activation == 'swish':
                layers.append(Swish(hidden_dim))
            elif activation == 'softplus':
                layers.append(nn.Softplus(hidden_dim, beta))
            if dropout:
                layers.append(nn.Dropout(0.5))
        
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
        
        if init:
            self.net.apply(init_weights)
         
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.net(x)
        out = out.squeeze()
        return out