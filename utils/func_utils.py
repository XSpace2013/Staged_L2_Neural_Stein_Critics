'''
Pulled from  "Learning the Stein Discrepancy for Training and 
Evaluating Energy-Based Models without Sampling" from
Grathwohl et al.
'''
import torch
import numpy as np
import random

# Utility Functions

def sample_batch(data, batch_size):
	all_inds = list(range(data.size(0)))
	chosen_inds = np.random.choice(all_inds, batch_size, replace=False)
	chosen_inds = torch.from_numpy(chosen_inds)
	return data[chosen_inds.long()]


def form_batches(data, batch_size):
	all_inds = list(range(data.size(0)))
	random.shuffle(all_inds)
	batches = []
	for i in range(0, len(all_inds), batch_size):
		batches.append(data[all_inds[i:i + batch_size]])
	return batches


def keep_grad(output, input, grad_outputs=None):
	return torch.autograd.grad(output, input,
		grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]


def exact_jacobian_trace(fx, x):
	vals = []
	for i in range(x.size(1)):
		fxi = fx[:, i]
		dfxi_dxi = keep_grad(fxi.sum(), x)[:, i][:, None]
		vals.append(dfxi_dxi)
	vals = torch.cat(vals, dim=1)
	return vals.sum(dim=1)


def approx_jacobian_trace(fx, x):
    eps = torch.randn_like(fx)
    eps_dfdx = keep_grad(fx, x, grad_outputs=eps)
    tr_dfdx = (eps_dfdx * eps).sum(-1)
    return tr_dfdx