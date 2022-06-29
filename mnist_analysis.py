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

class MNISTAnalyzer:
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
		self.n_test = args.n_test

		# network training parameters
		self.activation = args.activation
		self.nn_layers = args.nn_layers
		self.hidden_dim = args.hidden_dim
		self.n_epochs = args.n_epochs
		self.batch_size = args.batch_size
		self.lr = args.lr
		self.l2 = l2
		self.optimizer = args.optim

		# hypothesis test parameters
		self.alpha = args.alpha
		self.n_boot = args.n_boot

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

		# # first, generate validation pool and test pool for test statistics and null distributions
		if not os.path.exists("EBM_Training/validation_pool_embedded.pkl"):
			
			if not os.path.exists("EBM_Training/rbm_validation_pool.pkl"):
				with open("EBM_Training/rbm_validation_pool.pkl", "wb") as f:
					self.rbm_validation_pool = self.model_dist.sample(50000)
					pickle.dump(self.rbm_validation_pool.detach(), f)
			else:
				with open("EBM_Training/rbm_validation_pool.pkl", "rb") as f:
					self.rbm_validation_pool = pickle.load(f)

			if not os.path.exists("EBM_Training/validation_pool_indicators.pkl"):
				with open("EBM_Training/validation_pool.pkl", "wb") as f:
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
					
					self.validation_pool = torch.cat([self.rbm_validation_pool, test_digits_data, train_digits_data]).detach()
					pickle.dump(self.validation_pool.detach(), f)

				with open("EBM_Training/validation_pool_indicators.pkl", "wb") as f:
					# indicators : -1 means RBM, 1 means MNIST test digit, and -2 means MNIST train digit
					self.validation_pool_indicators = torch.cat([torch.ones(self.rbm_validation_pool.shape[0])*-1,
						torch.ones(test_digits_data.shape[0]),
						torch.ones(train_digits_data.shape[0])*-2])
					pickle.dump(self.validation_pool_indicators.detach(), f)
			else:
				with open("EBM_Training/validation_pool.pkl", "rb") as f:
					self.validation_pool = pickle.load(f)
				with open("EBM_Training/validation_pool_indicators.pkl", "rb") as f:
					self.validation_pool_indicators = pickle.load(f)

			with open("EBM_Training/validation_pool_embedded.pkl", "wb") as f:
				self.validation_pool_embedded = TSNE(n_components=2, init='random', learning_rate=50, perplexity=75, angle=0.2).fit_transform(self.validation_pool)
				pickle.dump(self.validation_pool_embedded, f)


		else:
			with open("EBM_Training/validation_pool.pkl", "rb") as f:
				self.validation_pool = pickle.load(f)
			with open("EBM_Training/validation_pool_indicators.pkl", "rb") as f:
				self.validation_pool_indicators = pickle.load(f)
			with open("EBM_Training/validation_pool_embedded.pkl", "rb") as f:
				self.validation_pool_embedded = pickle.load(f)

		validation_pool_generated_idx = (self.validation_pool_indicators == -1).nonzero().flatten()
		validation_pool_mnist_test_idx = (self.validation_pool_indicators == 1).nonzero().flatten()
		validation_pool_mnist_train_idx = (self.validation_pool_indicators == -2).nonzero().flatten()

		if not os.path.exists("EBM_Training/validation_pool_embedded_plot.png"):
			plt.figure(figsize=(9,7))
			plt.scatter(self.validation_pool_embedded[validation_pool_generated_idx,0],
				self.validation_pool_embedded[validation_pool_generated_idx,1], color='k', label="RBM Sample", s=15)
			plt.scatter(self.validation_pool_embedded[validation_pool_mnist_test_idx,0],
				self.validation_pool_embedded[validation_pool_mnist_test_idx,1], color='red', label="MNIST Digit 1", s=15)
			plt.scatter(self.validation_pool_embedded[validation_pool_mnist_train_idx,0],
				self.validation_pool_embedded[validation_pool_mnist_train_idx,1], color='red', s=15)
			plt.legend(fontsize=16)
			plt.tick_params(bottom=False,left=False)
			plt.gca().xaxis.set_ticklabels([])
			plt.gca().yaxis.set_ticklabels([])
			plt.savefig("EBM_Training/validation_pool_embedded_plot.png")
			plt.clf()
			plt.close()

		# determine exact number of digits which will be generated versus from true MNIST
		self.generated_test_size = np.sum([1 if random.random() < self.mixture_prop else 0 for _ in range(self.n_test)], dtype=int)
		self.true_test_size = self.n_test - self.generated_test_size

		# use random permutation of data from the generated test pool
		shuffled_generated_idx = validation_pool_generated_idx[torch.randperm(validation_pool_generated_idx.shape[0])]
		shuffled_mnist_test_idx = validation_pool_mnist_test_idx[torch.randperm(validation_pool_mnist_test_idx.shape[0])]

		true_test_idx = shuffled_mnist_test_idx[:self.true_test_size]
		generated_test_idx = shuffled_generated_idx[:self.generated_test_size]
		
		self.data_test = torch.cat([self.validation_pool[true_test_idx],
			self.validation_pool[generated_test_idx]]).requires_grad_().to(self.device)

		self.embedded_data = np.concatenate([self.validation_pool_embedded[true_test_idx],
			self.validation_pool_embedded[generated_test_idx]])

		self.targets_test = torch.ones(true_test_idx.shape[0])

		self.leftover_rbm_idx = shuffled_generated_idx[self.generated_test_size:]


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


	def test_gof(self):
		'''
		Given the true and model distributions, perform goodness-of-fit
		hypothesis testing using the learned Stein critic
		'''

		test_stats, _ = self.stein_discrepancy(self.data_test)
		test_stat = test_stats.mean()
		null_stats = torch.zeros(self.n_boot)
		for t in range(self.n_boot):
			null_idx = self.leftover_rbm_idx[torch.randperm(self.leftover_rbm_idx.shape[0])][:self.n_test]
			null_sample = (self.validation_pool[null_idx]).clone()
			null_sample = null_sample.requires_grad_()
			null_test_stats, _ = self.stein_discrepancy(null_sample)
			null_test_stat = null_test_stats.mean()
			null_stats[t] = null_test_stat.detach()
			if self.device == "cuda:0":
				torch.cuda.empty_cache()
		null_stats = null_stats.to(self.device)

		p = torch.sum(null_stats.cpu() > test_stat.cpu()) / self.n_boot
		if p < self.alpha:
			return 1, null_stats, test_stat
		else:
			return 0, null_stats, test_stat


	def analyze_critic(self, checkpoint, savedir):
		'''
		Given the true and model distributions, fit  the critic function
		with corresponding square integrability penalization factor
		and plot/report the results
		'''

		self.critic = MLP(in_dim=self.dim, out_dim=self.dim, hidden_dim=self.hidden_dim,
			nn_layers=self.nn_layers, activation=self.activation, dropout=False, init=True)
		self.critic.load_state_dict(torch.load(checkpoint))
		self.critic.to(self.device)
		
		_, null_stats, test_stat = self.test_gof()
		
		plt.figure(figsize=(9,7))
		plt.hist(np.array(null_stats.detach()), bins=20)
		plt.axvline(test_stat.detach(), color='r', ls='--', label='Test Statistic')
		plt.xlabel(r"$\hat{T}$",fontsize=28)
		plt.gca().yaxis.set_ticks([0,20,40,60])
		plt.gca().xaxis.set_ticks([0,100,200])
		plt.xticks(fontsize=24)
		plt.yticks(fontsize=24)
		plt.legend(fontsize=20)
		plt.tight_layout()
		plt.savefig(savedir+"/null_distribution.png")
		plt.clf()
		plt.close()

		sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
		plotter = lambda p, x: tv.utils.save_image(x.clamp(0, 1), p, normalize=False, nrow=sqrt(x.size(0)))
		

		# sample and plot the generated test images which have highest and lowest critic norms
		sample_critic_norms = torch.linalg.norm(self.critic(self.data_test[self.true_test_size:]), ord=2, dim=1)
		rbm_data_sd, _ = self.stein_discrepancy(self.data_test[self.true_test_size:])
		rbm_data_sd = rbm_data_sd.detach()
		
		percentile_90 = torch.quantile(sample_critic_norms, 0.9)
		percentile_10 = torch.quantile(sample_critic_norms, 0.1)
		
		_, largest_idx = torch.sort(rbm_data_sd, descending=True)
		most_wrong = self.data_test[self.true_test_size + largest_idx[:12]]
		most_wrong = most_wrong.view(most_wrong.size(0), 1, 28, 28)
		plotter(savedir+"/rbm_worst_images.png", torch.sigmoid(most_wrong.cpu()))

		_, smallest_idx = torch.sort(rbm_data_sd, descending=False)
		most_right = self.data_test[self.true_test_size + smallest_idx[:12]]
		most_right = most_right.view(most_right.size(0), 1, 28, 28)
		plotter(savedir+"/rbm_best_images.png", torch.sigmoid(most_right.cpu()))
		
		plt.figure(figsize=(15,5))
		plt.hist(np.array(sample_critic_norms.detach().flatten()), bins=100)
		plt.axvline(percentile_10.detach(), color='r', ls='--', label='0.1 Quantile')
		plt.axvline(percentile_90.detach(), color='r', ls='--', label='0.9 Quantile')
		plt.xlabel("Model Sample Norm",fontsize=14)
		plt.yscale('symlog')
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		plt.legend(fontsize=14)
		plt.savefig(savedir+"/model_hist.png")
		plt.clf()
		plt.close()


		# sample and plot the true MNIST test images which have highest and lowest critic norms
		test_critic_norms = torch.linalg.norm(self.critic(self.data_test[:self.true_test_size]), ord=2, dim=1)
		mnist_data_sd, _ = self.stein_discrepancy(self.data_test[:self.true_test_size])
		mnist_data_sd = mnist_data_sd.detach()
		
		percentile_90 = torch.quantile(test_critic_norms, 0.9)
		percentile_10 = torch.quantile(test_critic_norms, 0.1)
		
		plt.figure(figsize=(15,5))
		plt.hist(np.array(test_critic_norms.detach().flatten()), bins=100)
		plt.axvline(percentile_10.detach(), color='r', ls='--', label='0.1 Quantile')
		plt.axvline(percentile_90.detach(), color='r', ls='--', label='0.9 Quantile')
		plt.xlabel("True Sample Norm",fontsize=14)
		plt.yscale('symlog')
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		plt.legend(fontsize=14)
		plt.savefig(savedir+"/true_hist.png")
		plt.clf()
		plt.close()


		# map all test points to 2D, and visualize as heatmap according to their norm and if they are true MNIST samples
		all_mags = torch.cat([test_critic_norms.detach(), sample_critic_norms.detach()],dim=0)
		all_sds = torch.cat([mnist_data_sd, rbm_data_sd], dim=0)
		
		_, largest_idx = torch.sort(all_sds, descending=True)
		most_wrong = self.data_test[largest_idx[:12]]
		most_wrong = most_wrong.view(most_wrong.size(0), 1, 28, 28)
		plotter(savedir+"/worst_images.png", torch.sigmoid(most_wrong.cpu()))

		_, smallest_idx = torch.sort(all_sds, descending=False)
		most_right = self.data_test[smallest_idx[:12]]
		most_right = most_right.view(most_right.size(0), 1, 28, 28)
		plotter(savedir+"/best_images.png", torch.sigmoid(most_right.cpu()))
		

		all_mags = np.array(all_mags)
		all_sd, _ = self.stein_discrepancy(self.data_test)
		all_sd = all_sd.detach().numpy()
		plt.figure(figsize=(9,7))
		plt.scatter(list(range(1,all_mags.shape[0]+1)), all_mags, s=5, color='k')
		plt.scatter(list(range(1,test_critic_norms.shape[0]+1)), all_mags[:test_critic_norms.shape[0]],
			s=5, color='red', label='MNIST Data: {} points'.format(test_critic_norms.shape[0]))
		plt.axhline(np.mean(all_mags), c='r', label="Mean: {}".format(np.mean(all_mags)))
		plt.axhline(np.mean(all_mags[self.true_test_size:]), c='k', label="RBM Mean: {}".format(np.mean(all_mags[self.true_test_size:])))
		plt.legend(fontsize=14)
		plt.savefig(savedir+"/critic_applied_to_test.png")
		plt.clf()
		plt.close()

		plt.figure(figsize=(9,7))
		plt.scatter(list(range(1,all_sd.shape[0]+1)), all_sd, s=5, color='k')
		plt.scatter(list(range(1,test_critic_norms.shape[0]+1)), all_sd[:test_critic_norms.shape[0]],
			s=5, color='red', label='MNIST Data: {} points'.format(test_critic_norms.shape[0]))
		plt.axhline(np.mean(all_sd), c='r', label="Mean: {}".format(np.mean(all_sd)))
		plt.axhline(np.mean(all_sd) + np.std(all_sd),
			c='r', alpha=0.5, label="Std: {}".format(np.std(all_sd)))
		plt.axhline(np.mean(all_sd) - np.std(all_sd), c='k', alpha=0.5)
		plt.axhline(np.mean(all_sd[self.true_test_size:]), c='k', label="RBM Mean: {}".format(np.mean(all_sd[self.true_test_size:])))
		plt.legend(fontsize=14)
		plt.savefig(savedir+"/test_stats.png")
		plt.clf()
		plt.close()

		plt.figure(figsize=(9,7))
		plt.scatter(list(range(self.true_test_size+1,all_sd.shape[0]+1)), all_sd[self.true_test_size:], s=5, color='k')
		plt.axhline(np.mean(all_sd[self.true_test_size:]), c='k', label="RBM Mean: {}".format(np.mean(all_sd[self.true_test_size:])))
		plt.axhline(np.mean(all_sd[self.true_test_size:]) + np.std(all_sd[self.true_test_size:]),
			c='k', alpha=0.5, label="RBM Std: {}".format(np.std(all_sd[self.true_test_size:])))
		plt.axhline(np.mean(all_sd[self.true_test_size:]) - np.std(all_sd[self.true_test_size:]), c='k', alpha=0.5)
		plt.legend(fontsize=14)
		plt.savefig(savedir+"/rbm_test_stats.png")
		plt.clf()
		plt.close()


		null_idx = self.leftover_rbm_idx[torch.randperm(self.leftover_rbm_idx.shape[0])][:self.n_test]
		null_sample = (self.validation_pool[null_idx]).clone().requires_grad_()
		null_sd, _ = self.stein_discrepancy(null_sample)
		null_sd = null_sd.detach().numpy()
		null_mags = torch.linalg.norm(self.critic(null_sample), ord=2, dim=1).detach().numpy()
		plt.figure(figsize=(9,7))
		plt.scatter(list(range(1,null_mags.shape[0]+1)), null_mags, s=5, color='k')
		plt.axhline(np.mean(null_mags), c='k', label="Mean: {}".format(np.mean(null_mags)))
		plt.legend(fontsize=14)
		plt.savefig(savedir+"/critic_applied_to_null.png")
		plt.clf()
		plt.close()

		plt.figure(figsize=(9,7))
		plt.scatter(list(range(1,null_sd.shape[0]+1)), null_sd, s=5, color='k')
		plt.axhline(np.mean(null_sd), c='k', label="Mean: {}".format(np.mean(null_sd)))
		plt.axhline(np.mean(null_sd) + np.std(null_sd), c='k', alpha=0.5, label="Std: {}".format(np.std(null_sd)))
		plt.axhline(np.mean(null_sd) - np.std(null_sd), c='k', alpha=0.5)
		plt.legend(fontsize=14)
		plt.savefig(savedir+"/test_stats_null.png")
		plt.clf()
		plt.close()

		
		X_embedded = self.embedded_data
		embedded_true = self.embedded_data[:self.true_test_size]

		plt.figure(figsize=(9,7))
		plt.scatter(X_embedded[:,0], X_embedded[:,1], c = self.l2 * all_sd, cmap='OrRd', s=15)
		plt.tick_params(bottom=False,left=False)
		plt.gca().xaxis.set_ticklabels([])
		plt.gca().yaxis.set_ticklabels([])
		plt.colorbar()
		plt.savefig(savedir+"/embeddings_heat.png")
		plt.clf()
		plt.close()

		plt.figure(figsize=(9,7))
		colors= ['tab:red','tab:orange']
		digit_labels = [1]
		plt.scatter(X_embedded[self.true_test_size:,0], X_embedded[self.true_test_size:,1], color='k', label="RBM Sample", s=15)
		for i in range(len(digit_labels)):
			digit_idx = (self.targets_test == digit_labels[i])
			plt.scatter(embedded_true[digit_idx,0], embedded_true[digit_idx,1], color=colors[i], label="MNIST Digit {}".format(digit_labels[i]), s=15)
		plt.legend(fontsize=16)
		plt.tick_params(bottom=False,left=False)
		plt.gca().xaxis.set_ticklabels([])
		plt.gca().yaxis.set_ticklabels([])
		plt.savefig(savedir+"/embeddings_modelvtrue.png")
		plt.clf()
		plt.close()