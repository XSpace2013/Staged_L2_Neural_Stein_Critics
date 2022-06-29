'''
Based heavily upon (and largely borrowed from) 
"Learning the Stein Discrepancy for Training and 
Evaluating Energy-Based Models without Sampling" from
Grathwohl et al.
'''
import os
import torch
import torch.distributions as distributions
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import arrow
from utils.class_utils import *
from utils.func_utils import *
from utils.log_utils import *

class RegularizedTrainer:
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

		# true distribution
		self.dim = args.dim
		self.mean_pert = args.mean_pert
		self.std_pert = args.std_pert
		self.weight_pert = args.weight_pert
		self.generate_true_dist()

		# model distribution
		self.generate_model_dist()
		
		# generate data from the true distribution
		data = self.true_dist.sample(args.n_train + args.n_val).detach()
		self.data_train = data[:args.n_train].to(self.device)
		self.data_val = data[args.n_train:].requires_grad_().to(self.device)

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
		self.approx_support = 1.5
		self.x_range = None
		self.mse_sample = args.mse_sample


	def generate_true_dist(self):
		'''
		Initialize the true distribution and its parameters
		'''

		if self.dim == 1:
			self.data_means = torch.arange(-2+1,2,2.).to(self.device)
			self.data_scales = (torch.ones(2,)*1.).to(self.device)
			self.data_weights = torch.ones(2,).to(self.device)
			
			self.mix = distributions.Categorical(self.data_weights)
			self.comp = distributions.Normal(self.data_means, self.data_scales)
			self.true_dist = GaussianMixture(self.mix,self.comp).to(self.device)

		else:
			self.data_means = torch.stack([torch.zeros(self.dim), 0.5*torch.ones(self.dim)]).to(self.device)
			pert_cov1 = torch.eye(self.dim)
			pert_cov1[0,1] = self.std_pert
			pert_cov1[1,0] = self.std_pert
			pert_cov2 = torch.eye(self.dim)
			pert_cov2[0,1] = -self.std_pert
			pert_cov2[1,0] = -self.std_pert
			# D = torch.diag(torch.tensor([self.weight_pert, 1.])).to(self.device)
			D = torch.diag(torch.cat((torch.tensor([self.weight_pert]),torch.ones(self.dim-1)))).to(self.device)
			pert_cov2 = D @ pert_cov2 @ D
			# print(pert_cov1)
			# print(pert_cov2)
			self.data_scales = torch.stack([pert_cov1, pert_cov2]).to(self.device)
			self.data_weights = torch.ones(2,).to(self.device)

			self.mix = distributions.Categorical(self.data_weights)
			self.comp = distributions.MultivariateNormal(self.data_means, self.data_scales)
			self.true_dist = GaussianMixture(self.mix,self.comp, dim=self.dim).to(self.device)


	def generate_model_dist(self):
		'''
		Initialize the perturbed distribution and its parameters
		'''

		if self.dim == 1:
			self.model_means = torch.tensor([-self.mean_pert, 1.]).to(self.device)
			self.model_scales = (torch.tensor([1., self.std_pert])).to(self.device)
			self.model_weights = torch.ones(2,).to(self.device)

			self.model_mix = distributions.Categorical(self.model_weights)
			self.model_comp = distributions.Normal(self.model_means, self.model_scales)
			self.model_dist = GaussianMixture(self.model_mix, self.model_comp, dim=self.dim).to(self.device)

		else:
			self.model_means = self.data_means
			self.model_scales = torch.stack([torch.eye(self.dim), torch.eye(self.dim)]).to(self.device)
			self.model_weights = torch.ones(2,).to(self.device)
			
			self.model_mix = distributions.Categorical(self.model_weights)
			self.model_comp = distributions.MultivariateNormal(self.model_means, self.model_scales)
			self.model_dist = GaussianMixture(self.model_mix, self.model_comp, dim=self.dim).to(self.device)


	def optimal_critic(self, x, l2, grad=False):
		'''
		Given the target and model distributions, return the value at 
		x of the optimal critic function which maximizes Stein discrepancy
		according to square integrability penalization factor l2

		Parameters
		----------
			x : torch tensor
				points over which to evaluate the critic function
			l2 : float
				square integrability penalization factor
			grad : boolean
				specify whether or not to track gradients
		'''

		if not grad:
			x = torch.tensor(x).to(self.device)
			x.requires_grad_()
		log_diff = self.model_dist(x)-self.true_dist(x)
		result = torch.autograd.grad(log_diff.sum(), x, create_graph=True)[0]
		return (1/(2*l2)) * result


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
			norms : torch tensor
				norm of the Stein critic applied to x
		'''

		logq_u = self.model_dist(x)
		score_q = keep_grad(logq_u.sum(), x)
		
		fx = self.critic(x)
		if self.dim==1:
			fx = fx[:, None]

		sq_fx = (score_q * fx).sum(-1)
		
		if self.dim <= 2:
			tr_dfdx = exact_jacobian_trace(fx, x)
		else:
			tr_dfdx = approx_jacobian_trace(fx, x)
		
		norms = (fx * fx).sum(1)
		stats = (sq_fx + tr_dfdx)
		return stats, norms


	def plot_critic(self, this_l2, savedir, batch_num):
		'''
		Plot the given values of the current critic function compared to
		the values expected by the optimum critic

		Parameters
		----------
			savedir : string
				save directory for this plot
			batch_num : int
				the number of minibatches that have passed in training
		'''

		if self.dim == 1:

			x_range = np.linspace(-4,4,num=200)
			trained_critic = self.critic(torch.tensor(x_range, dtype=torch.float).to(self.device))
			
			fig = plt.figure(figsize=(9,7))

			plt.plot(x_range,trained_critic.cpu().detach(),color='k',label="Trained Critic")
			if this_l2 != 0:
				optimum_critic = self.optimal_critic(x_range, this_l2/2)
				plt.plot(x_range,optimum_critic.cpu().detach(),color='purple',ls='--',label="Optimal Critic")

			plt.title("LAM={:.2e}   Epoch {}".format(this_l2, batch_num // (self.data_train.shape[0]//self.batch_size)),fontsize=32)
			plt.ylim(1.1*torch.min(optimum_critic.cpu().detach()), 1.5*torch.max(optimum_critic.cpu().detach()))
			plt.yticks(fontsize=24)
			plt.legend(loc=3,fontsize=32)
			plt.tick_params(bottom=False)
			plt.gca().xaxis.set_ticklabels([])
			plt.grid()
			plt.ylim(-130, 55)
			plt.tight_layout()
			plt.savefig("{}/trained_batch{:03d}.png".format(savedir,batch_num))

			plt.clf()
			plt.close()


			fig = plt.figure(figsize=(9,7))
			plt.plot(x_range,this_l2*trained_critic.cpu().detach(),color='k',label="Scaleless Trained Critic")
			if this_l2 != 0:
				optimum_critic = this_l2*self.optimal_critic(x_range, this_l2/2)
				plt.plot(x_range,optimum_critic.cpu().detach(),color='r',ls='--',label="Scaleless Optimal Critic")

			plt.title("LAM={:.2e}   Epoch {}".format(this_l2, batch_num // (self.data_train.shape[0]//self.batch_size)),fontsize=32)
			plt.ylim(1.1*torch.min(optimum_critic.cpu().detach()), 1.5*torch.max(optimum_critic.cpu().detach()))
			plt.yticks(fontsize=24)
			plt.legend(loc=3,fontsize=32)
			plt.tick_params(bottom=False)
			plt.gca().xaxis.set_ticklabels([])
			plt.grid()
			plt.tight_layout()
			plt.savefig("{}/scaleless_batch{:03d}.png".format(savedir,batch_num))

			plt.clf()
			plt.close()

		elif self.dim == 2:
			x1, x2 = np.mgrid[-4.5:4.5:20*1j, -4.5:4.5:20*1j]
			x_range = np.vstack((x1.flatten(), x2.flatten())).T

			trained_critic = self.critic(torch.tensor(x_range, dtype=torch.float).to(self.device))

			fig = plt.figure(figsize=(9,7))

			plt.quiver(x_range[:,0],x_range[:,1],
				trained_critic.cpu().detach()[:,0],trained_critic.cpu().detach()[:,1],scale=25/float(this_l2),color='k')

			plt.title("LAM={:.2e}   Epoch {}".format(this_l2, batch_num // (self.data_train.shape[0]//self.batch_size)),fontsize=28)
			plt.tick_params(bottom=False,left=False)
			plt.gca().xaxis.set_ticklabels([])
			plt.gca().yaxis.set_ticklabels([])
			plt.tight_layout()
			plt.savefig("{}/trained_batch{:03d}.png".format(savedir,batch_num))

			plt.clf()
			plt.close()

			if batch_num == 0 and float(this_l2) != 0:
				optimum_critic = self.optimal_critic(torch.Tensor(x_range).to(self.device).requires_grad_(), float(this_l2)/2, grad=True)

				fig = plt.figure(figsize=(9,7))

				plt.quiver(x_range[:,0],x_range[:,1],
					optimum_critic.cpu().detach()[:,0],optimum_critic.cpu().detach()[:,1],scale=25/float(this_l2),color='r')

				plt.title("Optimal Critic",fontsize=28)
				plt.tick_params(bottom=False,left=False)
				plt.gca().xaxis.set_ticklabels([])
				plt.gca().yaxis.set_ticklabels([])
				plt.tight_layout()
				plt.savefig("{}/optimal_critic.png".format(savedir))

				plt.clf()
				plt.close()


	def get_diffs(self, l2, data=None):
		'''
		Given the fit and optimal critics, calculate the MSE
		under a sample from the model distribution
		'''

		if data is None:
			q_sample = self.model_dist.sample(self.mse_sample).detach()
		else:
			q_sample = data.detach()
		norms = torch.linalg.norm((l2/2)*self.critic(q_sample).detach().view(-1,self.dim) - (l2/2)*self.optimal_critic(np.array(q_sample.cpu()), l2/2).detach().view(-1,self.dim), ord=2, dim=1)
		return torch.pow(norms, 2).mean()


	def validation_mse(self, l2, validation_data):
		'''
		Given the fit and optimal critics, calculate the MSE
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

		if plot:
			assert savedir is not None
			if self.dim <= 2 and not os.path.exists(savedir+"/critics"):
				os.mkdir(savedir+"/critics")


		# create a wide MLP with selected activation for critic function
		self.critic = MLP(in_dim=self.dim, out_dim=self.dim, hidden_dim=self.hidden_dim,
			nn_layers=self.nn_layers, activation=self.activation, dropout=False, init=True).to(self.device)
		if self.optimizer == 'sgd':
			optimizer = optim.SGD(self.critic.parameters(), lr=self.lr, momentum=0.9)
		elif self.optimizer == 'adam':
			optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay=0)

		e_mse = []
		if type(self.l2) == LambdaStager:
			init_mse = self.get_diffs(self.l2.lams[0])
		else:
			init_mse = self.get_diffs(self.l2)
		e_mse.append(init_mse)

		num_batches = 0
		if plot and self.dim <= 2:
			if type(self.l2) == LambdaStager:
				self.plot_critic(self.l2.lams[0], savedir + "/critics", batch_num=num_batches)
			else:
				self.plot_critic(self.l2, savedir + "/critics", batch_num=num_batches)
		elif len(checkpoints) > 0 and self.dim <= 2:
			if type(self.l2) == LambdaStager or type(self.l2) == LambdaStagerAuto:
				self.plot_critic(self.l2.lams[0], savedir, batch_num=num_batches)
			else:
				self.plot_critic(self.l2, savedir, batch_num=num_batches)


		e_losses, e_stein_discs, e_penalties, e_val_stein_discs, e_validation_mse = [], [], [], [], []
		batch_losses, batch_stein_discs, batch_penalties = [], [], []
		batch_mse = []
		save_critic_times, batch_times, epoch_times = [], [], []
		for epoch in range(self.n_epochs):
			epoch_start_time = arrow.now()
			
			if type(self.l2) == LambdaStager:
				this_l2 = self.l2.get_lam()
			else:
				this_l2 = self.l2
			
			batch_size = self.batch_size
			this_lr	= self.lr

			x_batches = form_batches(self.data_train, batch_size)
			losses, stein_discs, penalties = [], [], []
			for x in x_batches:
				batch_start_time = arrow.now()
				self.critic.train()

				optimizer.zero_grad()
				x.requires_grad_()

				stats, norms = self.stein_discrepancy(x)
				mean, std = stats.mean(), stats.std()
				l2_penalty = norms.mean() * (this_l2/2)
				
				loss = -1. * mean + l2_penalty
				
				loss.backward()
				optimizer.step()

				batch_times.append((arrow.now()-batch_start_time).total_seconds())

				losses.append(loss.item())
				stein_discs.append(mean.item())
				penalties.append(l2_penalty.item())
				
				batch_losses.append(loss.item())
				batch_stein_discs.append(mean.item())
				batch_penalties.append(l2_penalty.item())

				if this_l2 != 0:
					train_mse = self.get_diffs(this_l2, data=x)
				else:
					train_mse = 0
				batch_mse.append(train_mse)
				
				num_batches += 1
				
				self.critic.eval()

				# plot the critic
				if plot and self.dim <= 2:
					self.plot_critic(this_l2, savedir + "/critics", batch_num=num_batches)

			# calculate Stein disc of val dataset
			optimizer.zero_grad()
			val_stats, _ = self.stein_discrepancy(self.data_val)
			e_val_stein_discs.append(val_stats.mean().item())

			e_validation_mse.append(self.validation_mse(this_l2, self.data_val))

			# append all metrics to lists
			e_losses.append(np.mean(losses))
			e_stein_discs.append(np.mean(stein_discs))
			e_penalties.append(np.mean(penalties))
			if this_l2 != 0:
				mse = self.get_diffs(this_l2)
			else:
				mse = e_mse[-1]
			e_mse.append(mse)

			if (epoch+1) in checkpoints:
				start_time = arrow.now()
				if this_l2 != 0 and self.dim <= 2:
					self.plot_critic(this_l2, savedir, batch_num=num_batches)
				torch.save(self.critic.state_dict(), "{}/critic_epoch_{}".format(savedir, epoch+1))
				save_critic_times.append((arrow.now()-start_time).total_seconds())

			self.critic.train()

			if epoch % self.report_freq == 0:
				print('Epoch %d:\tloss = %1.3e\tdisc = %1.3e\tbatch_size = %d\tlr = %1.3e\tlam = %1.1e'%(epoch, e_losses[-1], e_stein_discs[-1],
					batch_size, this_lr, this_l2))

			if type(self.l2) == LambdaStager:
				self.l2.update()

			epoch_times.append((arrow.now()-epoch_start_time).total_seconds())

		print('End Epoch:\t\tloss = %1.3e\tdisc = %1.3e\tbatch_size = %d\tlr = %1.3e\tlam = %1.1e'%(e_losses[-1], e_stein_discs[-1], batch_size, this_lr, this_l2))

		batch_stein_discs = torch.tensor(batch_stein_discs).detach().numpy()
		batch_losses = torch.tensor(batch_losses).detach().numpy()
		batch_penalties = torch.tensor(batch_penalties).detach().numpy()
		e_stein_discs = torch.tensor(e_stein_discs).detach().numpy()
		e_val_stein_discs = torch.tensor(e_val_stein_discs).detach().numpy()
		e_losses = torch.tensor(e_losses).detach().numpy()
		e_penalties = torch.tensor(e_penalties).detach().numpy()
		e_mse = torch.tensor(e_mse).detach().numpy()
		batch_mse = torch.tensor(batch_mse).detach().numpy()


		# visualize the trained critic or return results
		optimizer.zero_grad()
		self.critic.eval()
		
		if plot:
			batch_plotter([self.l2], batch_stein_discs.reshape(1,-1), 0, "Training Stein Discrepancy", "{}/batch_stein_traj.png".format(savedir))
			batch_plotter([self.l2], batch_losses.reshape(1,-1), 0, "Training Loss", "{}/batch_loss_traj.png".format(savedir))
			batch_plotter([self.l2], batch_penalties.reshape(1,-1), 0, "Training Norm Regularization", "{}/batch_regularization_traj.png".format(savedir))

			batch_plotter([self.l2], batch_mse.reshape(1,-1), 0, "Training MSE", "{}/training_mse.png".format(savedir))

			epoch_plotter([self.l2], e_stein_discs.reshape(1,-1), "Training Stein Discrepancy", "{}/stein_traj.png".format(savedir))
			epoch_plotter([self.l2], e_val_stein_discs.reshape(1,-1), "Validation Stein Discrepancy", "{}/pop_stein_traj.png".format(savedir))
			epoch_plotter([self.l2], e_losses.reshape(1,-1), "Training Loss", "{}/loss_traj.png".format(savedir))
			epoch_plotter([self.l2], e_penalties.reshape(1,-1), "Training Norm Regularization", "{}/regularization_traj.png".format(savedir))
			epoch_plotter([self.l2], e_mse.reshape(1,-1), "MSE", "{}/mse_traj.png".format(savedir))

			np.save("{}/stein_discs.npy".format(savedir), e_stein_discs)
			np.save("{}/validation_stein_discs.npy".format(savedir), e_val_stein_discs)
			np.save("{}/losses.npy".format(savedir), e_losses)
			np.save("{}/penalties.npy".format(savedir), e_penalties)
			np.save("{}/mse.npy".format(savedir), e_mse)
			np.save("{}/train_mse.npy".format(savedir), batch_mse)

		if compare_dir is not None:
			return e_stein_discs, e_val_stein_discs, e_losses, e_penalties, e_mse, batch_mse
		
		elif len(checkpoints) > 0:
			if type(self.l2) == LambdaStager:
				epoch_plotter([self.l2], [e_mse], "MSE", "{}/mse_traj.png".format(savedir))
				epoch_plotter([self.l2], [e_val_stein_discs], "Validation Stein Discrepancy", "{}/val_sd_traj.png".format(savedir))
				epoch_plotter([self.l2], [e_validation_mse], "Validation MSE", "{}/validation_mse_traj.png".format(savedir))
			else:
				epoch_to_batches_plotter([e_mse], "MSE", "{}/mse_traj.png".format(savedir), self.data_train.shape[0] // self.batch_size)
				epoch_to_batches_plotter([e_val_stein_discs], "Validation Stein Discrepancy", "{}/val_sd_traj.png".format(savedir), self.data_train.shape[0] // self.batch_size)
				epoch_to_batches_plotter([e_validation_mse], "Validation MSE", "{}/validation_mse_traj.png".format(savedir), self.data_train.shape[0] // self.batch_size)
			return e_mse, batch_mse, e_val_stein_discs, e_validation_mse, save_critic_times, epoch_times, batch_times


	def fit_critic_efficient(self, savedir, checkpoints=[]):

		self.critic = MLP(in_dim=self.dim, out_dim=self.dim, hidden_dim=self.hidden_dim,
			nn_layers=self.nn_layers, activation=self.activation, dropout=False, init=True).to(self.device)
		if self.optimizer == 'sgd':
			optimizer = optim.SGD(self.critic.parameters(), lr=self.lr, momentum=0.9)
		elif self.optimizer == 'adam':
			optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay=0)

		e_validation_mse = []
		save_critic_times, form_batch_times, compute_valid_times = [], [], []
		batch_times, epoch_times = [], []
		for epoch in range(self.n_epochs):
			epoch_start_time = arrow.now()
			if type(self.l2) == LambdaStager:
				this_l2 = self.l2.get_lam()
			else:
				this_l2 = self.l2
			batch_size = self.batch_size
			this_lr	= self.lr

			form_batch_start = arrow.now()
			x_batches = form_batches(self.data_train, batch_size)
			form_batch_times.append((arrow.now()-form_batch_start).total_seconds())
			for x in x_batches:
				batch_start_time = arrow.now()
				self.critic.train()

				optimizer.zero_grad()
				x.requires_grad_()

				stats, norms = self.stein_discrepancy(x)
				mean, std = stats.mean(), stats.std()
				l2_penalty = norms.mean() * (this_l2/2)
				
				loss = -1. * mean + l2_penalty
				
				loss.backward()
				optimizer.step()

				batch_times.append((arrow.now()-batch_start_time).total_seconds())
				
			self.critic.eval()

			# calculate Stein disc of val dataset
			optimizer.zero_grad()

			valid_start = arrow.now()
			e_validation_mse.append(self.validation_mse(this_l2, self.data_val))
			compute_valid_times.append((arrow.now()-valid_start).total_seconds())

			if (epoch+1) in checkpoints:
				start_time = arrow.now()
				torch.save(self.critic.state_dict(), "{}/critic_epoch_{}".format(savedir, epoch+1))
				save_critic_times.append((arrow.now()-start_time).total_seconds())

			if type(self.l2) == LambdaStager:
				self.l2.update()

			epoch_times.append((arrow.now()-epoch_start_time).total_seconds())

		return e_validation_mse, save_critic_times, form_batch_times, compute_valid_times, epoch_times, batch_times