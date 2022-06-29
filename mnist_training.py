'''
Based heavily upon (and largely borrowed from) 
"Learning the Stein Discrepancy for Training and 
Evaluating Energy-Based Models without Sampling" from
Grathwohl et al.
'''
import torch
import torch.distributions as distributions
import torch.optim as optim
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle
import arrow
from utils.class_utils import *
from utils.func_utils import *
from utils.log_utils import *

class MNISTRegularizedTrainer:
	"""
	A class to analyze loss over training
	for a critic function distinguishing a
	model distribution from a sample.
	"""
	def __init__(self, args, device, l2):
		"""
		Initialize the true and model distributions, and generate
		the sample from the true distribution

		Parameters
		----------
			args : argparser arguments
				command line arguments specifying all parameters
			device : torch device
				GPU or CPU to run calculations
			l2 : float or LambdaStager
				regularization or staging scheme
		"""

		self.device = device

		# EBM parameters
		self.dim = 784
		self.rbm_hidden_dim = 100
		self.burn_in = 2000
		self.rbm = True

		# data sampling parameters
		self.mixture_prop = args.mix_prop
		self.n_train = args.n_train
		self.n_test = args.n_val

		# network training parameters
		self.activation = args.activation
		self.nn_layers = args.nn_layers
		self.hidden_dim = args.hidden_dim
		self.n_epochs = args.n_epochs
		self.batch_size = args.batch_size
		self.lr = args.lr
		self.l2 = l2
		self.optimizer = args.optim

		# logger parameters
		self.report_freq = args.report_freq

		# data and model initialization
		self.loader_batch_size = 256
		self.dload_train, self.dload_test, self.dset_train, self.dset_test = self.get_data()
		self.load_model()
		self.assign_data()


	def load_model(self):
		"""
		Load the pre-trained EBM of MNIST digits
		"""

		for x, _ in self.dload_train:
			init_batch = x.view(x.size(0), -1)
			break

		if self.rbm:
			B = torch.randn((self.dim, self.rbm_hidden_dim)) / self.rbm_hidden_dim
			c = torch.randn((1, self.rbm_hidden_dim))
			b = init_batch.mean(0)[None, :]
			self.model_dist = GaussianBernoulliRBM(B, b, c, burn_in=self.burn_in)
		else:
			net = MLP(in_dim=self.dim, out_dim=1, hidden_dim=self.rbm_hidden_dim)
			mu, std = init_batch.mean(), init_batch.std() + 1.
			base_dist = distributions.Normal(mu, std)
			self.model_dist = EBM(net, base_dist, learn_base_dist=True)

		ckpt = torch.load("EBM_Training/checkpt.pth")
		self.model_dist.load_state_dict(ckpt['ebm_state_dict'])


	def assign_data(self):
		"""
		Load data from the EBM. First, collect pools of generated
		and true data from MNIST. Then, determine exact amount of
		true and generated data to use in train and test set. Finally,
		shuffle the indices of the generated pool and real data, and
		assign data to the training and testing datasets.
		"""

		if not os.path.exists("EBM_Training/test_pool.pkl"):
			with open("EBM_Training/test_pool.pkl", "wb") as f:
				self.model_test_pool = self.model_dist.sample(30000)
				pickle.dump(self.model_test_pool.detach(), f)
		else:
			with open("EBM_Training/test_pool.pkl", "rb") as f:
				self.model_test_pool = pickle.load(f)

		# collect all train data from MNIST, extracting only specific digits
		full_train_data = []
		full_train_targets = []
		for _, (x, y) in enumerate(self.dload_train):
			full_train_data.append(x)
			full_train_targets.append(y)
		full_train_data = torch.cat(full_train_data)
		full_train_data = full_train_data.view(full_train_data.size(0), -1)
		full_train_targets = torch.cat(full_train_targets)

		train_digits_idx = (full_train_targets == 1).nonzero().flatten()
		train_digits_data = full_train_data[train_digits_idx]
		train_digits_targets = full_train_targets[train_digits_idx]

		# determine exact number of digits which will be generated versus from true MNIST
		generated_train_size = np.sum([1 if random.random() < self.mixture_prop else 0 for _ in range(self.n_train)], dtype=int)
		true_train_size = self.n_train - generated_train_size

		# collect all test data from MNIST, extracting only specific digits
		full_test_data = []
		full_test_targets = []
		for _, (x, y) in enumerate(self.dload_test):
			full_test_data.append(x)
			full_test_targets.append(y)
		full_test_data = torch.cat(full_test_data)
		full_test_data = full_test_data.view(full_test_data.size(0), -1)
		full_test_targets = torch.cat(full_test_targets)

		test_digits_idx = (full_test_targets == 1).nonzero().flatten()
		test_digits_data = full_test_data[test_digits_idx]
		test_digits_targets = full_test_targets[test_digits_idx]

		# determine exact number of digits which will be generated versus from true MNIST
		self.generated_test_size = np.sum([1 if random.random() < self.mixture_prop else 0 for _ in range(self.n_test)], dtype=int)
		self.true_test_size = self.n_test - self.generated_test_size

		# use random permutation of data from the test pool
		generated_idx = torch.randperm(self.model_test_pool.shape[0])
		
		# training/testing data includes true MNIST digits and generated digits
		true_train_idx = torch.randperm(train_digits_data.shape[0])[:true_train_size]
		generated_train_idx = generated_idx[:generated_train_size]
		self.data_train = torch.cat([train_digits_data[true_train_idx],
			self.model_test_pool[generated_train_idx]]).detach().to(self.device)
		self.targets_train = train_digits_targets[true_train_idx]

		true_test_idx = torch.randperm(test_digits_data.shape[0])[:self.true_test_size]
		generated_test_idx = generated_idx[generated_train_size:generated_train_size+self.generated_test_size]
		self.data_test = torch.cat([test_digits_data[true_test_idx],
			self.model_test_pool[generated_test_idx]]).requires_grad_().to(self.device)
		self.targets_test = test_digits_targets[true_test_idx]

		self.model_test_pool_leftover_idx = generated_idx[generated_train_size+self.generated_test_size:]


	def logit(self, x, alpha=1e-6):
		x = x * (1 - 2 * alpha) + alpha
		return torch.log(x) - torch.log(1 - x)


	def middle_transform(self, x):
		return x * (255. / 256.) + (torch.rand_like(x) / 256.)


	def get_data(self):
		transform = tr.Compose([tr.ToTensor(), self.middle_transform, self.logit])
		
		dset_train = tv.datasets.MNIST(root="data", train=True, transform=transform, download=True)
		dset_test = tv.datasets.MNIST(root="data", train=False, transform=transform, download=True)

		dload_train = DataLoader(dset_train, batch_size=self.loader_batch_size, shuffle=True, num_workers=4, drop_last=True)
		dload_test = DataLoader(dset_test, batch_size=self.loader_batch_size, shuffle=True, num_workers=4, drop_last=True)
		return dload_train, dload_test, dset_train, dset_test


	def stein_discrepancy(self, x):
		'''
		Given the current state of the critic function, return the Stein
		discrepancy estimated over batch x

		Parameters
		----------
			x : torch tensor
				points over which to evaluate the function

		Returns
		-------
			discrepancies : torch tensor
				values of the quantity inside the expected value
				of the Stein discrepancy evaluated at x
		'''

		if self.rbm:
			sq = self.model_dist.score_function(x)
			lp = None
		else:
			logp_u = distribution(x, lp=False)
			sq = keep_grad(logp_u.sum(), x)

		fx = self.critic(x)
		sq_fx = (sq * fx).sum(-1)
		
		tr_dfdx = approx_jacobian_trace(fx, x)
		
		stats = sq_fx + tr_dfdx
		norms = (fx * fx).sum(1)
		return stats, norms


	def validation_mse(self, l2, validation_data):
		'''
		Given the fit and optimal critics, calculate the MAE
		under a sample from the true distribution
		'''

		stats, norms = self.stein_discrepancy(validation_data)
		mean = stats.mean()
		l2_penalty = norms.mean() * (l2/2)

		return (2. * l2 * (-1. * mean + l2_penalty)).detach()


	def fit_critic(self, plot=True, savedir=None, compare_dir=None, checkpoints=[]):
		'''
		Given the true and model distributions, fit  the critic function
		with corresponding square integrability penalization factor
		and plot/report the results
		'''

		# create a wide MLP with selected activation for critic function
		self.critic = MLP(in_dim=self.dim, out_dim=self.dim, hidden_dim=self.hidden_dim,
			nn_layers=self.nn_layers, activation=self.activation, dropout=False, init=True).to(self.device)
		if self.optimizer == 'sgd':
			optimizer = optim.SGD(self.critic.parameters(), lr=self.lr, momentum=0.9)
		elif self.optimizer == 'adam':
			optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay=0)

		e_losses, e_stein_discs, e_penalties, e_test_stein_discs, e_validation_mse, e_test_sd_ratios = [], [], [], [], [], []
		batch_losses, batch_stein_discs, batch_penalties = [], [], []
		num_batches = 0
		for epoch in range(self.n_epochs):

			if type(self.l2) == LambdaStager:
				this_l2 = self.l2.get_lam()
			else:
				this_l2 = self.l2

			batch_size = self.batch_size
			this_lr	= self.lr

			x_batches = form_batches(self.data_train, self.batch_size)
			losses, stein_discs, penalties = [], [], []
			for x in x_batches:

				self.critic.train()

				optimizer.zero_grad()
				x.requires_grad_()

				stats, norms = self.stein_discrepancy(x)
				mean, std = stats.mean(), stats.std()
				l2_penalty = norms.mean() * (this_l2/2)
				
				loss = -1. * mean + l2_penalty
				
				loss.backward()
				optimizer.step()

				losses.append(loss.item())
				stein_discs.append(mean.item())
				penalties.append(l2_penalty.item())
				
				batch_losses.append(loss.item())
				batch_stein_discs.append(mean.item())
				batch_penalties.append(l2_penalty.item())
				
				num_batches += 1
				
				self.critic.eval()


			# calculate Stein disc of test dataset
			optimizer.zero_grad()
			test_stats, _ = self.stein_discrepancy(self.data_test)
			e_test_stein_discs.append(test_stats.mean().item())

			null_idx = self.model_test_pool_leftover_idx[torch.randperm(self.model_test_pool_leftover_idx.shape[0])][:self.n_test]
			null_sample = (self.model_test_pool[null_idx]).clone().requires_grad_()
			null_sd, _ = self.stein_discrepancy(null_sample)
			e_test_sd_ratios.append(torch.sqrt(torch.tensor(self.n_test)) * test_stats.mean().item() / (test_stats.std().item() + null_sd.std().item()))

			e_validation_mse.append(self.validation_mse(this_l2, self.data_test))
			
			# append all metrics to lists
			e_losses.append(np.mean(losses))
			e_stein_discs.append(np.mean(stein_discs))
			e_penalties.append(np.mean(penalties))

			if (epoch+1) in checkpoints:
				torch.save(self.critic.state_dict(), "{}/critic_epoch_{}".format(savedir, epoch+1))

			self.critic.train()

			if epoch % self.report_freq == 0:
				print('Epoch %d:\tloss = %1.3e\tdisc = %1.3e\tlam = %1.1e'%(epoch, e_losses[-1], e_stein_discs[-1], this_l2))

			if type(self.l2) == LambdaStager:
				self.l2.update()

		print('End Epoch:\t\tloss = %1.3e\tdisc = %1.3e\tbatch_size = %d\tlr = %1.3e\tlam = %1.1e'%(e_losses[-1], e_stein_discs[-1], batch_size, this_lr, this_l2))

		batch_stein_discs = torch.tensor(batch_stein_discs).detach().numpy()
		batch_losses = torch.tensor(batch_losses).detach().numpy()
		batch_penalties = torch.tensor(batch_penalties).detach().numpy()
		e_stein_discs = torch.tensor(e_stein_discs).detach().numpy()
		e_test_stein_discs = torch.tensor(e_test_stein_discs).detach().numpy()
		e_test_sd_ratios = torch.tensor(e_test_sd_ratios).detach().numpy()
		e_losses = torch.tensor(e_losses).detach().numpy()
		e_penalties = torch.tensor(e_penalties).detach().numpy()

		# visualize the trained critic or return results
		optimizer.zero_grad()
		self.critic.eval()

		if plot:
			batch_plotter([self.l2], batch_stein_discs.reshape(1,-1), 0, "Training Stein Discrepancy", "{}/batch_stein_traj.png".format(savedir))
			batch_plotter([self.l2], batch_losses.reshape(1,-1), 0, "Training Loss", "{}/batch_loss_traj.png".format(savedir))
			batch_plotter([self.l2], batch_penalties.reshape(1,-1), 0, "Training Norm Regularization", "{}/batch_regularization_traj.png".format(savedir))

			epoch_plotter([self.l2], e_stein_discs.reshape(1,-1), "Training Stein Discrepancy", "{}/stein_traj.png".format(savedir))
			epoch_plotter([self.l2], e_test_stein_discs.reshape(1,-1), "Validation Stein Discrepancy", "{}/pop_stein_traj.png".format(savedir))
			epoch_plotter([self.l2], e_losses.reshape(1,-1), "Training Loss", "{}/loss_traj.png".format(savedir))
			epoch_plotter([self.l2], e_penalties.reshape(1,-1), "Training Norm Regularization", "{}/regularization_traj.png".format(savedir))

			np.save("{}/stein_discs.npy".format(savedir), e_stein_discs)
			np.save("{}/validation_stein_discs.npy".format(savedir), e_test_stein_discs)
			np.save("{}/losses.npy".format(savedir), e_losses)
			np.save("{}/penalties.npy".format(savedir), e_penalties)


		if compare_dir is not None:
			return e_stein_discs, e_test_stein_discs, e_losses, e_penalties, None, None

		elif len(checkpoints) > 0:
			if type(self.l2) == LambdaStager:
				epoch_plotter([self.l2], [e_test_stein_discs], "Validation Stein Discrepancy", "{}/test_sd_traj.png".format(savedir))
				epoch_plotter([self.l2], [e_stein_discs], "Validation Stein Discrepancy", "{}/train_sd_traj.png".format(savedir))
				epoch_plotter([self.l2], [e_test_sd_ratios], "Testing Stein Discrepancy Ratio", "{}/test_sd_ratio_traj.png".format(savedir))
				epoch_plotter([self.l2], [e_stein_discs-e_test_stein_discs], "Train-Validation Stein Discrepancy", "{}/sd_diff_traj.png".format(savedir))
				epoch_plotter([self.l2], [e_validation_mse], "Validation MSE", "{}/validation_mse_traj.png".format(savedir))
			else:
				epoch_to_batches_plotter([e_test_stein_discs], "Validation Stein Discrepancy", "{}/test_sd_traj.png".format(savedir), self.data_train.shape[0] // self.batch_size)
				epoch_to_batches_plotter([e_stein_discs], "Validation Stein Discrepancy", "{}/train_sd_traj.png".format(savedir), self.data_train.shape[0] // self.batch_size)
				epoch_to_batches_plotter([e_test_sd_ratios], "Testing Stein Discrepancy Ratio", "{}/test_sd_ratio_traj.png".format(savedir), self.data_train.shape[0] // self.batch_size)
				epoch_to_batches_plotter([e_stein_discs-e_test_stein_discs], "Train-Validation Stein Discrepancy", "{}/sd_diff_traj.png".format(savedir), self.data_train.shape[0] // self.batch_size)
				epoch_to_batches_plotter([e_validation_mse], "Validation MSE", "{}/validation_mse_traj.png".format(savedir), self.data_train.shape[0] // self.batch_size)
			return e_test_stein_discs, e_validation_mse, e_test_sd_ratios, e_stein_discs