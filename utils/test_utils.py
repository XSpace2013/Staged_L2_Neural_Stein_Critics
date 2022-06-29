'''
Pulled from 
"A Kernel Test of Goodness of Fit" from
Chwialkowski et al.
'''
from scipy.spatial.distance import squareform, pdist
import arrow
import numpy as np


class _KSD:
	def __init__(self, scaling):
		self.scaling = scaling

	def kernel_matrix(self, X):

		# check for stupid mistake
		assert X.shape[0] > X.shape[1]

		sq_dists = squareform(pdist(X, 'sqeuclidean'))

		K = np.exp(-sq_dists/ self.scaling)
		return K

	def gradient_k_wrt_x(self, X, K, dim):

		X_dim = X[:, dim]
		assert X_dim.ndim == 1

		differences = X_dim.reshape(len(X_dim), 1) - X_dim.reshape(1, len(X_dim))

		return -2.0 / self.scaling * K * differences

	def gradient_k_wrt_y(self, X, K, dim):
		return -self.gradient_k_wrt_x(X, K, dim)

	def second_derivative_k(self, X, K, dim):
		X_dim = X[:, dim]
		assert X_dim.ndim == 1

		differences = X_dim.reshape(len(X_dim), 1) - X_dim.reshape(1, len(X_dim))

		sq_differences = differences ** 2

		return 2.0 * K * (self.scaling - 2 * sq_differences) / self.scaling ** 2

	def get_statistic_multiple_dim(self, samples, log_pdf_gradients, dim):
		num_samples = len(samples)

		this_log_pdf_gradients = log_pdf_gradients[:, dim]
		K = self.kernel_matrix(samples)
		gradient_k_x = self.gradient_k_wrt_x(samples, K, dim)
		gradient_k_y = self.gradient_k_wrt_y(samples, K, dim)
		second_derivative = self.second_derivative_k(samples, K, dim)

		# use broadcasting to mimic the element wise looped call
		pairwise_log_gradients = this_log_pdf_gradients.reshape(num_samples, 1) \
								 * this_log_pdf_gradients.reshape(1, num_samples)
		A = pairwise_log_gradients * K

		B = gradient_k_x * this_log_pdf_gradients
		C = (gradient_k_y.T * this_log_pdf_gradients).T
		D = second_derivative

		V_statistic = A + B + C + D

		stat = num_samples * np.mean(V_statistic)
		return V_statistic, stat

	def compute_pvalues_for_processes(self, U_matrix, chane_prob, num_bootstrapped_stats=100):
		N = U_matrix.shape[0]
		bootsraped_stats = np.zeros(num_bootstrapped_stats)

		for proc in range(num_bootstrapped_stats):
			# W = np.sign(orsetinW[:,proc])
			W = simulatepm(N, chane_prob)
			WW = np.outer(W, W)
			st = np.mean(U_matrix * WW)
			bootsraped_stats[proc] = N * st

		stat = N * np.mean(U_matrix)

		return float(np.sum(bootsraped_stats > stat)) / num_bootstrapped_stats


class KSD:
	def __init__(self, model_dist, scaling):
		self.tester = _KSD(scaling)
		self.model_dist = model_dist

	def is_from_null(self, alpha, samples, chane_prob, n_boot, start_time):
		dims = samples.shape[1]
		boots = n_boot # 10 * int(dims / alpha)
		num_samples = samples.shape[0]

		samples = samples.requires_grad_()
		log_probs = self.model_dist(samples)
		log_probs.sum().backward()
		log_pdf_gradients = np.array(samples.grad)
		samples = samples.detach().numpy()

		U = np.zeros((num_samples, num_samples))
		for dim in range(dims):
			U2, _ = self.tester.get_statistic_multiple_dim(samples, log_pdf_gradients, dim)
			U += U2

		test_stat_duration = (arrow.now()-start_time).total_seconds()


		start_time = arrow.now()
		
		p = self.tester.compute_pvalues_for_processes(U, chane_prob, boots)
		
		boot_duration = (arrow.now()-start_time).total_seconds()

		return p, test_stat_duration, boot_duration


def simulatepm(N, p_change):
	'''
	:param N:
	:param p_change:
	:return:
	'''
	X = np.zeros(N) - 1
	change_sign = np.random.rand(N) < p_change
	for i in range(N):
		if change_sign[i]:
			X[i] = -X[i - 1]
		else:
			X[i] = X[i - 1]
	return X