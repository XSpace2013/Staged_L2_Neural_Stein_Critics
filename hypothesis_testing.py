'''
Based heavily upon (and largely borrowed from) 
"Learning the Stein Discrepancy for Training and 
Evaluating Energy-Based Models without Sampling" from
Grathwohl et al.
'''
import torch
import torch.distributions as distributions
import arrow
from utils.class_utils import *
from utils.func_utils import *
from utils.log_utils import *
from utils.test_utils import KSD

class HypothesisTester:
	"""
	A class to analyze loss over training
	for a critic function distinguishing a
	model distribution from a sample.
	"""
	def __init__(self, args, device):
		"""
		Initialize the true and model distributions, and generate
		the sample from the true distribution

		Parameters
		----------
			args : argparser arguments
				command line arguments specifying all parameters
			device : torch device
				GPU or CPU to run calculations
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
		self.data_test = self.true_dist.sample(args.n_test).requires_grad_().to(self.device)

		# network training parameters
		self.activation = args.activation
		self.nn_layers = args.nn_layers
		self.hidden_dim = args.hidden_dim
		
		# hypothesis test parameters
		self.alpha = args.alpha
		self.n_runs = args.n_runs
		self.n_boot = args.n_boot


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
			D = torch.diag(torch.cat((torch.tensor([self.weight_pert]),torch.ones(self.dim-1)))).to(self.device)
			pert_cov2 = D @ pert_cov2 @ D
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


	def test_gof(self):
		'''
		Given the true and model distributions, perform goodness-of-fit
		hypothesis testing using the learned Stein critic
		'''

		n_reject = 0
		test_stat_durations, boot_durations = [], []
		for i in range(self.n_runs):
			start_time = arrow.now()
			
			test_stats, _ = self.stein_discrepancy(self.true_dist.sample(self.data_test.shape[0]).requires_grad_().to(self.device))
			test_stat = test_stats.mean()
			
			test_stat_durations.append((arrow.now()-start_time).total_seconds())

			
			start_time = arrow.now()

			null_pool = self.model_dist.sample(self.data_test.shape[0]*3).requires_grad_().to(self.device)
			null_stat_pool, _ = self.stein_discrepancy(null_pool)
			null_stat_pool = null_stat_pool.detach()
			
			n_repeat = int(torch.ceil(torch.tensor((self.data_test.shape[0] * self.n_boot) / null_stat_pool.shape[0])))
			full_null_stat_pool = null_stat_pool.repeat(n_repeat)
			shuffled_null_stat_pool = full_null_stat_pool[torch.randperm(full_null_stat_pool.shape[0])][:(self.n_boot*self.data_test.shape[0])]
			
			null_stats = shuffled_null_stat_pool.view(self.n_boot, self.data_test.shape[0]).mean(dim=1)
			null_stats = null_stats.to(self.device)

			p = torch.sum(null_stats.cpu() > test_stat.cpu()) / self.n_boot
			
			if p < self.alpha:
				n_reject += 1

			boot_durations.append((arrow.now() - start_time).total_seconds())

			if (i%50) == 0:
				print("Run {} | Power: {:.2e} | Duration: {:.2e} s".format(i, n_reject/(i+1),
					test_stat_durations[-1] + boot_durations[-1]))

		print("End Power: {:.2e} | Avg Duration: {:.2e} s".format(n_reject/self.n_runs, np.mean(test_stat_durations) + np.mean(boot_durations)))
		return n_reject / self.n_runs, np.mean(test_stat_durations), np.mean(boot_durations)


	def test_gof_ksd(self):
		'''
		Given the true and model distributions, perform goodness-of-fit
		hypothesis testing using the kernel Stein discrepancy
		'''

		n_reject = 0
		test_stat_durations, boot_durations = [], []
		for i in range(self.n_runs):
			start_time = arrow.now()
			
			data_test = self.true_dist.sample(self.data_test.shape[0]).requires_grad_().to(self.device)
			dist_mat = torch.cdist(data_test, data_test, p=2)
			rbf_bw = float(dist_mat.flatten()[dist_mat.triu().flatten().nonzero()].median())
			rbf_bw = 2 * (rbf_bw**2)
			KSD_tester = KSD(self.model_dist, rbf_bw)

			p, test_stat_duration, boot_duration = KSD_tester.is_from_null(self.alpha, data_test, 0.1, self.n_boot, start_time)
			if p < self.alpha:
				n_reject += 1

			test_stat_durations.append(test_stat_duration)
			boot_durations.append(boot_duration)
			
			if (i%50) == 0:
				print("Run {} | Power: {:.2e} | Duration: {:.2e} s | bw {:.2e}".format(i, n_reject/(i+1),
					test_stat_durations[-1] + boot_durations[-1], rbf_bw))
		
		print("End Power: {:.2e} | Avg Duration: {:.2e} s".format(n_reject/self.n_runs, np.mean(test_stat_durations) + np.mean(boot_durations)))
		return n_reject / self.n_runs, np.mean(test_stat_durations), np.mean(boot_durations)


	def get_power(self, checkpoint=None):
		'''
		Load MLP with selected activation for critic function
		and return the test power over n_runs trials
		'''

		if checkpoint is not None:
			self.critic = MLP(in_dim=self.dim, out_dim=self.dim, hidden_dim=self.hidden_dim,
				nn_layers=self.nn_layers, activation=self.activation, dropout=False, init=True)
			self.critic.load_state_dict(torch.load(checkpoint))
			self.critic.to(self.device)

			return self.test_gof()

		else:
			return self.test_gof_ksd()