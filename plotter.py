import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import arrow
from utils.class_utils import LambdaStager
import utils.log_utils as log_utils

parser = argparse.ArgumentParser()


# main selection
parser.add_argument('--procedure', type=str, default='compare_metrics', choices=['validation_epoch', 'neural_power', 'ksd_power', 'compare_metrics',
	'compare_power', 'compare_ksd', 'compare_ksd_best_bw', 'compare_ksd_bw', 'compare_ksd_bw_multiple'])


# model parameters
parser.add_argument('--dim', type=int, default=1)
parser.add_argument('--mean_pert', type=float, default=1.)
parser.add_argument('--std_pert', type=float, default=1.)
parser.add_argument('--weight_pert', type=float, default=1.)


# MNIST parameters
parser.add_argument('--mnist', action='store_true')
parser.add_argument('--mix_prop', type=float, default=0.97)


parser.add_argument('--n_train', type=int, default=80)
parser.add_argument('--n_val', type=int, default=20)
parser.add_argument('--n_test', type=int, default=100)


# regularization and staging parameters
parser.add_argument('--l2', type=float, default=0.01)
parser.add_argument('--stage_lambda', action='store_true')
parser.add_argument('--lam_init', type=float, default=5e-1)
parser.add_argument('--beta', type=float, default=0.85)
parser.add_argument('--lam_term', type=float, default=5e-3)


# hypothesis testing parameters
parser.add_argument('--model_instances', type=int, default=10)
parser.add_argument('--n_runs', type=int, default=500)
parser.add_argument('--n_boot', type=int, default=500)
parser.add_argument('--stop_epoch', type=int, default=45)

parser.add_argument('--bw_factor', type=float, default=1.)


parser.set_defaults(feature=False)


if __name__ == "__main__":
	args = parser.parse_args()

	n_sample = args.n_train + args.n_val + args.n_test

	if args.mnist:
		base_dir = "results/mnist/prop{}".format(args.mix_prop)
	else:
		if args.dim <= 1:
			base_dir = "results/dim_{}/mean{}_std{}".format(args.dim, args.mean_pert, args.std_pert)
		else:
			base_dir = "results/dim_{}/weight{}_std{}".format(args.dim, args.weight_pert, args.std_pert)




	## REGULARIZATION COMPARISON PARAMETERS ##

	color_list = ['tab:blue','tab:orange','tab:green', 'cyan', 'yellow', 'chartreuse', 'darkred', 'peru', 'indigo','tab:purple', 'teal']

	## 2 D case ##
	if args.dim == 2:
		lams = [1e-3, 1e-2, 1e-1, 1e0]
		pow_epochs = [60, 50, 30, 20, 30, 45]
	
	## 10 D case ##
	if args.dim == 10:
		lams = [2.5e-4, 1e-3, 4e-3, 1.6e-2, 6.4e-2, 2.56e-1, 1.024e0]
		pow_epochs = [50, 50, 45, 30, 20, 12, 12, 50, 45]

	## 25 D case ##
	if args.dim == 25:
		lams = [2.5e-4, 1e-3, 4e-3, 1.6e-2, 6.4e-2]
		pow_epochs = [40, 35, 30, 20, 20, 45, 40, 55]

	## MNIST case ##
	if args.mnist:
		lams = [1e-3, 1e-2]
		pow_epochs = []

	if args.stage_lambda:
		if args.dim == 2:
			lams.append((1.0, 0.90))
			lams.append((1.0, 0.95))

		if args.dim == 10:
			lams.append((0.5, 0.80))
			lams.append((0.5, 0.85))

		if args.dim == 25:
			lams.append((0.4, 0.80))
			lams.append((0.4, 0.85))
			lams.append((0.4, 0.90))

		if args.mnist:
			lams.append((0.5, 0.90))




	if args.procedure == 'validation_epoch':
		if args.stage_lambda:
			result_dir = "{}/critics_{}replicas_STAGELAM{:.1f}_beta{}_{}sample".format(base_dir, args.model_instances, args.lam_init, args.beta, n_sample)
		else:
			result_dir = "{}/critics_{}replicas_LAM{}_{}sample".format(base_dir, args.model_instances, args.l2, n_sample)

		start = arrow.now()

		stop_epochs = []

		for j in range(args.model_instances):

			validation_mse_vals = np.load("{}/critic_{}/validation_mse.npy".format(result_dir, j))
			if args.stage_lambda:
				stop_idx = np.argmin(validation_mse_vals[20:])
				stop_idx += 21
			else:
				stop_idx = np.argmin(validation_mse_vals)
				stop_idx += 1
			
			stop_epochs.append(stop_idx)

		valid_time = (arrow.now()-start).total_seconds()
		np.save("{}/valid_time.npy".format(result_dir), np.array(valid_time))

		print("Time: {:.2e}".format(valid_time))
		print("Validation epoch: {:.1f}".format(np.mean(stop_epochs)))




	elif args.procedure == 'neural_power':
		if args.stage_lambda:
			result_dir = "{}/critics_{}replicas_STAGELAM{:.1f}_beta{}_{}sample".format(base_dir, args.model_instances, args.lam_init, args.beta, n_sample)
		else:
			result_dir = "{}/critics_{}replicas_LAM{}_{}sample".format(base_dir, args.model_instances, args.l2, n_sample)

		pows = []
		test_stat_durations, boot_durations, train_durations, save_critic_durations = [], [], [], []
		total_durations = []

		for j in range(args.model_instances):
			pows.append(np.load("{}/critic_{}/power_epoch{:03d}_{}runs_{}boot.npy".format(result_dir, j, args.stop_epoch, args.n_runs, args.n_boot))[0])
			test_stat_durations.append(np.load("{}/critic_{}/test_stat_duration_epoch{:03d}_{}runs_{}boot.npy".format(result_dir, j, args.stop_epoch, args.n_runs, args.n_boot))[0])
			boot_durations.append(np.load("{}/critic_{}/boot_duration_epoch{:03d}_{}runs_{}boot.npy".format(result_dir, j, args.stop_epoch, args.n_runs, args.n_boot))[0])

			train_durations.append(np.load("{}/critic_{}/training_duration.npy".format(result_dir, j)))
			save_critic_durations.append(np.load("{}/critic_{}/save_critic_duration.npy".format(result_dir, j)))
			total_durations.append(float(train_durations[-1]) + test_stat_durations[-1] + boot_durations[-1])
		
		pow_mean = np.mean(pows)
		pow_std = np.std(pows)

		duration_mean = np.mean(total_durations)
		duration_std = np.std(total_durations)

		train_duration_mean = np.mean(train_durations)
		train_duration_std = np.std(train_durations)

		save_duration_mean = np.mean(save_critic_durations)
		save_duration_std = np.std(save_critic_durations)

		test_stat_duration_mean = np.mean(test_stat_durations)
		test_stat_duration_std = np.std(test_stat_durations)

		boot_duration_mean = np.mean(boot_durations)
		boot_duration_std = np.std(boot_durations)

		np.save(result_dir+"/mean_power_epoch{}_{}runs_{}boot.npy".format(args.stop_epoch, args.n_runs, args.n_boot), pow_mean)
		np.save(result_dir+"/std_power_epoch{}_{}runs_{}boot.npy".format(args.stop_epoch, args.n_runs, args.n_boot), pow_std)

		np.save(result_dir+"/mean_duration_epoch{}_{}runs_{}boot.npy".format(args.stop_epoch, args.n_runs, args.n_boot), duration_mean)
		np.save(result_dir+"/std_duration_epoch{}_{}runs_{}boot.npy".format(args.stop_epoch, args.n_runs, args.n_boot), duration_std)

		np.save(result_dir+"/mean_train_duration.npy", train_duration_mean)
		np.save(result_dir+"/std_train_duration.npy", train_duration_std)

		np.save(result_dir+"/mean_save_duration.npy", save_duration_mean)
		np.save(result_dir+"/std_save_duration.npy", save_duration_std)

		np.save(result_dir+"/mean_test_stat_duration_epoch{}_{}runs_{}boot.npy".format(args.stop_epoch, args.n_runs, args.n_boot), test_stat_duration_mean)
		np.save(result_dir+"/std_test_stat_duration_epoch{}_{}runs_{}boot.npy".format(args.stop_epoch, args.n_runs, args.n_boot), test_stat_duration_std)

		np.save(result_dir+"/mean_boot_duration_epoch{}_{}runs_{}boot.npy".format(args.stop_epoch, args.n_runs, args.n_boot), boot_duration_mean)
		np.save(result_dir+"/std_boot_duration_epoch{}_{}runs_{}boot.npy".format(args.stop_epoch, args.n_runs, args.n_boot), boot_duration_std)

		print("Neural Stein Mean Power: {:.3f} ".format(pow_mean) + "stdev: {:.3e}".format(pow_std))
		print("Neural Stein Mean Total Duration: {:.3e} ".format(duration_mean) + "stdev: {:.3e}".format(duration_std))
		
		print("\nNeural Stein Mean Train Duration: {:.3e} ".format(train_duration_mean) + "stdev: {:.3e}".format(train_duration_std))
		print("Neural Stein Mean Save Critic Duration: {:.3e} ".format(save_duration_mean) + "stdev: {:.3e}".format(save_duration_std))
		print("Neural Stein Mean Test Stat Compute Duration: {:.3e} ".format(test_stat_duration_mean) + "stdev: {:.3e}".format(test_stat_duration_std))
		print("Neural Stein Mean Bootstrap Duration: {:.3e} ".format(boot_duration_mean) + "stdev: {:.3e}".format(boot_duration_std))




	elif args.procedure == 'ksd_power':
		if args.bw_factor == 1.:
			test_savedir = "{}/KSD_results/{}runs_{}boot_{}test".format(base_dir, args.n_runs, args.n_boot, args.n_test)
		else:
			test_savedir = "{}/KSD_results/{}runs_{}boot_{}test_{}bw".format(base_dir, args.n_runs, args.n_boot, args.n_test, args.bw_factor)

		pows, test_stat_durations, boot_durations = [], [], []
		for j in range(args.model_instances):
			pows.append(np.load("{}/sample_{}/power.npy".format(test_savedir, j)))
			test_stat_durations.append(np.load("{}/sample_{}/test_stat_duration.npy".format(test_savedir, j)))
			boot_durations.append(np.load("{}/sample_{}/boot_duration.npy".format(test_savedir, j)))

		pow_mean = np.mean(pows)
		pow_std = np.std(pows)

		duration_mean = np.mean(np.array(test_stat_durations)+np.array(boot_durations))
		duration_std = np.std(np.array(test_stat_durations)+np.array(boot_durations))

		test_stat_duration_mean = np.mean(test_stat_durations)
		test_stat_duration_std = np.std(test_stat_durations)

		boot_duration_mean = np.mean(boot_durations)
		boot_duration_std = np.std(boot_durations)

		np.save(test_savedir+"/mean_power.npy", pow_mean)
		np.save(test_savedir+"/std_power.npy", pow_std)

		np.save(test_savedir+"/mean_duration.npy", duration_mean)
		np.save(test_savedir+"/std_duration.npy", duration_std)

		np.save(test_savedir+"/mean_test_stat_duration.npy", test_stat_duration_mean)
		np.save(test_savedir+"/std_test_stat_duration.npy", test_stat_duration_std)

		np.save(test_savedir+"/mean_boot_duration.npy", boot_duration_mean)
		np.save(test_savedir+"/std_boot_duration.npy", boot_duration_std)

		print("KSD Mean Power: {:.3f} ".format(pow_mean) + "stdev: {:.3e}".format(pow_std))
		print("KSD Mean Total Duration: {:.3e} ".format(duration_mean) + "stdev: {:.2e}".format(duration_std))
		print("\nKSD Mean Test Stat Duration: {:.3e} ".format(test_stat_duration_mean) + "stdev: {:.2e}".format(test_stat_duration_std))
		print("KSD Mean Boot Duration: {:.3e} ".format(boot_duration_mean) + "stdev: {:.2e}".format(boot_duration_std))




	elif args.procedure == 'compare_metrics':
		lam_stage_list, lam_stage_methods = [], []
		lam_mse_means, lam_mse_stds, lam_test_sd_ratio_means, lam_test_sd_ratio_stds = [], [], [], []
		lam_valid_mse_means, lam_valid_mse_stds = [], []
		lam_validation_sd_means, lam_validation_sd_stds = [], []

		for i in range(len(lams)):
			if type(lams[i]) == tuple:
				result_dir = "{}/critics_{}replicas_STAGELAM{:.1f}_beta{}_{}sample".format(base_dir, args.model_instances, lams[i][0], lams[i][1], n_sample)
				lam_stage_list.append(np.load(result_dir+"/critic_0/staged_lams.npy"))
				lam_stage_methods.append(r'$\Lambda$' + "({:.2f}".format(lams[i][1]) + ")")
			
			else:
				result_dir = "{}/critics_{}replicas_LAM{}_{}sample".format(base_dir, args.model_instances, lams[i], n_sample)
				lam_stage_methods.append("{:.1e}".format(lams[i]))
			
			
			mse_vals, validation_mse_vals, validation_sd_vals, test_sd_ratios = [], [], [], []
			for j in range(args.model_instances):
				if not args.mnist:
					mse_vals.append(np.load("{}/critic_{}/mse.npy".format(result_dir, j)))
				else:
					test_sd_ratios.append(np.load("{}/critic_{}/test_sd_ratio.npy".format(result_dir, j)))
				validation_mse_vals.append(np.load("{}/critic_{}/validation_mse.npy".format(result_dir, j)))
				validation_sd_vals.append(np.load("{}/critic_{}/validation_stein_discs.npy".format(result_dir, j)))
			
			
			if not args.mnist:
				lam_mse_means.append(np.array(mse_vals).mean(axis=0))
				lam_mse_stds.append(np.array(mse_vals).std(axis=0))
			else:
				lam_test_sd_ratio_means.append(np.array(test_sd_ratios).mean(axis=0))
				lam_test_sd_ratio_stds.append(np.array(test_sd_ratios).std(axis=0))
			

			lam_valid_mse_means.append(np.array(validation_mse_vals).mean(axis=0))
			lam_valid_mse_stds.append(np.array(validation_mse_vals).std(axis=0))

			
			if type(lams[i]) != tuple:
				lam_validation_sd_means.append((lams[i]*np.array(validation_sd_vals)).mean(axis=0))
				lam_validation_sd_stds.append((lams[i]*np.array(validation_sd_vals)).std(axis=0))
			else:
				new_validation_sd_vals = []
				validation_sd_vals = np.array(validation_sd_vals)
				for k in range(validation_sd_vals.shape[1]):
					new_validation_sd_vals.append(validation_sd_vals[:,k] * lam_stage_list[-1][k])
				lam_validation_sd_means.append(np.array(new_validation_sd_vals).mean(axis=1))
				lam_validation_sd_stds.append(np.array(new_validation_sd_vals).std(axis=1))


		if not args.mnist:
			fig = plt.figure(figsize=(9,7), dpi=300)
			for i in range(len(lams)):
				
				plt.plot(list(range(0,len(lam_mse_means[i]))), lam_mse_means[i], color=color_list[i],
					label=lam_stage_methods[i], ls = '--' if type(lams[i]) == tuple else '-')
				
				plt.fill_between(list(range(0,len(lam_mse_means[i]))), lam_mse_means[i] - lam_mse_stds[i],
					lam_mse_means[i] + lam_mse_stds[i],
					color=color_list[i], alpha=0.1)
			plt.ylabel(r"$\widehat{\rm MSE}_q$",fontsize=28)
			plt.xlabel("Training Epochs",fontsize=28)
			plt.yscale('linear')
			if args.dim == 2:
				plt.title("2 Dimension", fontsize=28)
				plt.gca().yaxis.set_ticks([0.0, 0.1])
				plt.legend(fontsize=20, loc=1)
			elif args.dim == 10:
				plt.title("10 Dimension", fontsize=28)
				plt.ylim(0.075,.25)
				plt.gca().yaxis.set_ticks([0.1, 0.2])
				plt.legend(fontsize=20, loc=1)
			elif args.dim == 25:
				plt.title("25 Dimension", fontsize=28)
				plt.ylim(0.165,.35)
				plt.gca().yaxis.set_ticks([0.2, 0.3])
				plt.legend(fontsize=20, loc=3)
			plt.gca().xaxis.set_ticks([0,20,40,60])
			plt.xticks(fontsize=24)
			plt.yticks(fontsize=24)
			plt.grid()
			plt.tight_layout()
			if args.stage_lambda:
				plt.savefig("{}/compare_mse_{}replicas_STAGELAM.png".format(base_dir, args.model_instances))
			else:
				plt.savefig("{}/compare_mse_{}replicas_LAM{}-{}.png".format(base_dir, args.model_instances, lams[0], lams[-1]))
			plt.clf()
			plt.close()


		if args.mnist:
			fig = plt.figure(figsize=(9,7), dpi=300)
			for i in range(len(lams)):
				plt.plot(list(range(1,len(lam_test_sd_ratio_means[i])+1)), lam_test_sd_ratio_means[i], color=color_list[i],
					label=lam_stage_methods[i], ls = '--' if type(lams[i]) == tuple else '-')
				
				plt.fill_between(list(range(1,len(lam_test_sd_ratio_means[i])+1)), lam_test_sd_ratio_means[i] - lam_test_sd_ratio_stds[i],
					lam_test_sd_ratio_means[i] + lam_test_sd_ratio_stds[i],
					color=color_list[i], alpha=0.1)
			plt.ylabel(r"$\hat{P}$",fontsize=28)
			plt.xlabel("Training Epochs",fontsize=28)
			plt.yscale('linear')
			plt.gca().yaxis.set_ticks([0,2,4])
			plt.xticks(fontsize=24)
			plt.yticks(fontsize=24)
			plt.legend(fontsize=20)
			plt.grid()
			plt.tight_layout()
			if args.stage_lambda:
				plt.savefig("{}/compare_test_sd_ratio_{}replicas_STAGELAM.png".format(base_dir, args.model_instances))
			else:
				plt.savefig("{}/compare_test_sd_ratio_{}replicas_LAM{}-{}.png".format(base_dir, args.model_instances, lams[0], lams[-1]))
			plt.clf()
			plt.close()


		fig = plt.figure(figsize=(9,7), dpi=300)
		for i in range(len(lams)):
			plt.plot(list(range(1,len(lam_valid_mse_means[i])+1)), lam_valid_mse_means[i], color=color_list[i],
				label=lam_stage_methods[i], ls = '--' if type(lams[i]) == tuple else '-')
			
			plt.fill_between(list(range(1,len(lam_valid_mse_means[i])+1)), lam_valid_mse_means[i] - lam_valid_mse_stds[i],
				lam_valid_mse_means[i] + lam_valid_mse_stds[i],
				color=color_list[i], alpha=0.1)
		plt.ylabel("Validation MSE",fontsize=28)
		plt.xlabel("Training Epochs",fontsize=28)
		plt.yscale('linear')
		if args.dim == 2:
			plt.title("2 Dimension", fontsize=28)
		elif args.dim == 10:
			plt.title("10 Dimension", fontsize=28)
		elif args.dim == 25:
			plt.title("25 Dimension", fontsize=28)
			plt.ylim(-0.35,.55)
		if args.mnist:
			plt.ylim(-5,20)
		plt.xticks(fontsize=24)
		plt.yticks(fontsize=24)
		plt.legend(fontsize=20)
		plt.tight_layout()
		if args.stage_lambda:
			plt.savefig("{}/compare_valid_mse_{}replicas_STAGELAM.png".format(base_dir, args.model_instances))
		else:
			plt.savefig("{}/compare_valid_mse_{}replicas_LAM{}-{}.png".format(base_dir, args.model_instances, lams[0], lams[-1]))
		plt.clf()
		plt.close()


		fig = plt.figure(figsize=(9,7), dpi=300)
		for i in range(len(lams)):
			plt.plot(list(range(0,len(lam_validation_sd_means[i]))), lam_validation_sd_means[i], color=color_list[i],
				label=lam_stage_methods[i], ls = '--' if type(lams[i]) == tuple else '-')
			
			plt.fill_between(list(range(0,len(lam_validation_sd_means[i]))), lam_validation_sd_means[i] - lam_validation_sd_stds[i], lam_validation_sd_means[i] + lam_validation_sd_stds[i],
				color=color_list[i], alpha=0.1)
		plt.ylabel("Validation Stein Discrepancy",fontsize=28)
		plt.xlabel("Training Epochs",fontsize=28)
		plt.yscale('symlog')
		if args.dim == 2:
			plt.title("2 Dimension", fontsize=28)
		elif args.dim == 10:
			plt.title("10 Dimension", fontsize=28)
		elif args.dim == 25:
			plt.title("25 Dimension", fontsize=28)
			plt.ylim(-0.1,1.5)
			plt.gca().yaxis.set_ticks([-1e-1, 1e-1, 1e0])
		plt.xticks(fontsize=24)
		plt.yticks(fontsize=24)
		plt.legend(fontsize=20)
		plt.grid()
		plt.tight_layout()
		if args.stage_lambda:
			plt.savefig("{}/compare_validation_sd_{}replicas_STAGELAM.png".format(base_dir, args.model_instances))
		else:
			plt.savefig("{}/compare_validation_sd_{}replicas_LAM{}-{}.png".format(base_dir, args.model_instances, lams[0], lams[-1]))
		plt.clf()
		plt.close()




	elif args.procedure == 'compare_power':
		lam_stage_list = []
		pow_means, pow_stds = [], []
		
		for i in range(len(lams)):
			if type(lams[i]) == tuple:
				result_dir = "{}/critics_{}replicas_STAGELAM{:.1f}_beta{}_{}sample".format(base_dir, args.model_instances, lams[i][0], lams[i][1], n_sample)
				lam_stage_list.append(np.load(result_dir+"/critic_0/staged_lams.npy"))
			
			else:
				result_dir = "{}/critics_{}replicas_LAM{}_{}sample".format(base_dir, args.model_instances, lams[i], n_sample)
			

			pow_means.append(np.load("{}/mean_power_epoch{}_{}runs_{}boot.npy".format(result_dir, pow_epochs[i], args.n_runs, args.n_boot)))
			pow_stds.append(np.load("{}/std_power_epoch{}_{}runs_{}boot.npy".format(result_dir, pow_epochs[i], args.n_runs, args.n_boot)))


		fig = plt.figure(figsize=(9,7), dpi=300)
		stage_counter = 0
		for i in range(len(lams)):
			if type(lams[i])==tuple:
				if lams[i] == (1.0, 0.90) or lams[i] == (0.5, 0.80) or lams[i] == (0.4, 0.85):
					plt.plot(lam_stage_list[stage_counter], pow_means[i] * np.ones(len(lam_stage_list[stage_counter])),
						c='red', label='Staged', lw=5, ls='--')
					plt.fill_between(lam_stage_list[stage_counter], pow_means[i] * np.ones(len(lam_stage_list[stage_counter])) - pow_stds[i],
						pow_means[i] * np.ones(len(lam_stage_list[stage_counter])) + pow_stds[i], color='red', alpha=0.1)
				stage_counter+=1
			else:
				plt.scatter(lams[i], pow_means[i], color='k')
				plt.errorbar(lams[i], pow_means[i], yerr=pow_stds[i], c='k', fmt="o")

		plt.ylabel("Testing Power",fontsize=28)
		plt.xlabel(r"$\lambda$",fontsize=28)
		if args.dim == 2:
			plt.title("2 Dimension", fontsize=28)
		elif args.dim == 10:
			plt.title("10 Dimension", fontsize=28)
		elif args.dim == 25:
			plt.title("25 Dimension", fontsize=28)
		plt.gca().yaxis.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0])
		plt.yscale('linear')
		plt.xscale('log')
		plt.xticks(fontsize=24)
		plt.yticks(fontsize=24)
		plt.ylim(0.15,1.05)
		if args.stage_lambda:
			plt.legend(fontsize=20)
		plt.tight_layout()
		plt.grid()
		if args.stage_lambda:
			plt.savefig("{}/compare_power_{}replicas_STAGELAM.png".format(base_dir, args.model_instances))
		else:
			plt.savefig("{}/compare_power_{}replicas_LAM{}-{}.png".format(base_dir, args.model_instances, lams[0], lams[-1]))
		plt.clf()
		plt.close()




	elif args.procedure == 'compare_ksd':
		ksd_mean_durations, ksd_std_durations = [], []
		ksd_mean_pows, ksd_std_pows = [], []
		ksd_mean_boot_durations, ksd_mean_stat_durations = [], []
		ksd_std_boot_durations, ksd_std_stat_durations = [], []

		neural_mean_durations, neural_std_durations = [], []
		neural_mean_pows, neural_std_pows = [], []
		neural_train_durations, neural_boot_durations, neural_stat_durations = [], [], []
		neural_train_durations_std, neural_boot_durations_std, neural_stat_durations_std = [], [], []
		neural_valid_durations = []

		ksd_dir = '{}/KSD_results/{}runs_'.format(base_dir, args.n_runs)
		if args.stage_lambda:
			neural_dir = '{}/critics_{}replicas_STAGELAM{:.1f}_beta{}'.format(base_dir, args.model_instances, args.lam_init, args.beta)
		else:
			neural_dir = '{}/critics_{}replicas_LAM{}'.format(base_dir, args.model_instances, args.l2)

		sample_sizes = [100, 200, 300, 500, 1000]
		if args.dim == 25 and args.stage_lambda:
			selected_epochs = [40, 40, 40, 35, 35]

		for i in range(len(sample_sizes)):

			ksd_mean_durations.append(np.load("{}{}boot_{}test/mean_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
			ksd_std_durations.append(np.load("{}{}boot_{}test/std_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
			
			ksd_mean_boot_durations.append(np.load("{}{}boot_{}test/mean_boot_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
			ksd_mean_stat_durations.append(np.load("{}{}boot_{}test/mean_test_stat_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
			
			ksd_std_boot_durations.append(np.load("{}{}boot_{}test/std_boot_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
			ksd_std_stat_durations.append(np.load("{}{}boot_{}test/std_test_stat_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
			
			ksd_mean_pows.append(np.load("{}{}boot_{}test/mean_power.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
			ksd_std_pows.append(np.load("{}{}boot_{}test/std_power.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
			

			neural_sample_dir = '{}_{}sample'.format(neural_dir, sample_sizes[i])
			
			neural_mean_durations.append(np.load("{}/mean_duration_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))
			neural_std_durations.append(np.load("{}/std_duration_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))

			neural_train_durations.append(np.load("{}/mean_train_duration.npy".format(neural_sample_dir)))
			neural_valid_durations.append(np.load("{}/valid_time.npy".format(neural_sample_dir)))
			neural_boot_durations.append(np.load("{}/mean_boot_duration_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))
			neural_stat_durations.append(np.load("{}/mean_test_stat_duration_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))
			
			neural_train_durations_std.append(np.load("{}/std_train_duration.npy".format(neural_sample_dir)))
			neural_boot_durations_std.append(np.load("{}/std_boot_duration_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))
			neural_stat_durations_std.append(np.load("{}/std_test_stat_duration_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))
			
			neural_mean_pows.append(np.load("{}/mean_power_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))
			neural_std_pows.append(np.load("{}/std_power_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))


		plt.figure(figsize=(9,7), facecolor='white', dpi=300)

		plt.errorbar(sample_sizes, neural_mean_pows, yerr=neural_std_pows, c='k', marker="x")
		plt.plot(sample_sizes, neural_mean_pows, c='k', label='Neural')

		plt.errorbar(sample_sizes, ksd_mean_pows, yerr=ksd_std_pows, c='r', marker="x")
		plt.plot(sample_sizes, ksd_mean_pows, c='r', label='KSD')

		plt.title('Power Comparison',fontsize=28)
		plt.xlabel("Sample Size",fontsize=28)
		plt.ylabel("Testing Power",fontsize=28)
		plt.xticks(fontsize=24)
		plt.yticks(fontsize=24)
		plt.grid()
		plt.legend(fontsize=24)
		plt.tight_layout()
		if args.stage_lambda:
			plt.savefig("{}/compare_ksd_power_STAGELAM.png".format(base_dir, args.model_instances))
		else:
			plt.savefig("{}/compare_ksd_power_LAM{}.png".format(base_dir, args.model_instances, args.l2))
		plt.clf()
		plt.close()


		plt.figure(figsize=(9,7), facecolor='white', dpi=300)

		neural_whole_time_mean = np.array(neural_train_durations)+np.array(neural_stat_durations)+np.array(neural_boot_durations)+np.array(neural_valid_durations)
		neural_whole_time_std = np.power(np.power(np.array(neural_train_durations_std), 2) + np.power(np.array(neural_stat_durations_std), 2) + np.power(np.array(neural_boot_durations_std), 2), 0.5)

		plt.errorbar(sample_sizes, neural_whole_time_mean, yerr=neural_whole_time_std, c='k', marker="x")
		plt.plot(sample_sizes, neural_whole_time_mean, c='k', label='Neural')

		ksd_whole_time_mean = np.array(ksd_mean_stat_durations)+np.array(ksd_mean_boot_durations)
		ksd_whole_time_std = np.power(np.power(np.array(ksd_std_stat_durations), 2) + np.power(np.array(ksd_std_boot_durations), 2), 0.5)

		plt.errorbar(sample_sizes, ksd_whole_time_mean, yerr=ksd_whole_time_std, c='r', marker="x")
		plt.plot(sample_sizes, ksd_whole_time_mean, c='r', label='KSD')

		plt.title('Total Time Comparison',fontsize=28)
		plt.xlabel("Sample Size",fontsize=28)
		plt.ylabel("Time (s)",fontsize=28)
		plt.xticks(fontsize=24)
		plt.yticks(fontsize=24)
		plt.grid()
		plt.legend(fontsize=24)
		plt.tight_layout()
		if args.stage_lambda:
			plt.savefig("{}/compare_ksd_time_STAGELAM.png".format(base_dir, args.model_instances))
		else:
			plt.savefig("{}/compare_ksd_time_LAM{}.png".format(base_dir, args.model_instances, args.l2))
		plt.clf()
		plt.close()


		plt.figure(figsize=(9,7), facecolor='white', dpi=300)

		neural_whole_test_stat_mean = np.array(neural_train_durations)+np.array(neural_stat_durations)+np.array(neural_valid_durations)
		neural_whole_test_stat_std = np.power(np.power(np.array(neural_train_durations_std), 2) + np.power(np.array(neural_stat_durations_std), 2), 0.5)

		plt.errorbar(sample_sizes, neural_whole_test_stat_mean, yerr=neural_whole_test_stat_std, c='k', marker="x")
		plt.plot(sample_sizes, neural_whole_test_stat_mean, ls='-', c='k', label='Neural Test Statistic')

		plt.errorbar(sample_sizes, neural_boot_durations, yerr=neural_boot_durations_std, c='k', ls='--', marker="x")
		plt.plot(sample_sizes, neural_boot_durations, c='k', ls='--', label='Neural Bootstrap')

		plt.errorbar(sample_sizes, ksd_mean_stat_durations, yerr=ksd_std_stat_durations, c='r', marker="x")
		plt.plot(sample_sizes, ksd_mean_stat_durations, c='r', ls='-', label='KSD Test Statistic')

		plt.errorbar(sample_sizes, ksd_mean_boot_durations, yerr=ksd_std_boot_durations, c='r', ls='--', marker="x")
		plt.plot(sample_sizes, ksd_mean_boot_durations, c='r', ls='--', label='KSD Bootstrap')

		plt.title('Breakdown Time Comparison',fontsize=28)
		plt.xlabel("Sample Size",fontsize=28)
		plt.ylabel("Time (s)",fontsize=28)
		plt.xticks(fontsize=24)
		plt.yticks(fontsize=24)
		plt.grid()
		plt.legend(fontsize=24)
		plt.tight_layout()
		if args.stage_lambda:
			plt.savefig("{}/compare_ksd_time_breakdown_STAGELAM.png".format(base_dir, args.model_instances))
		else:
			plt.savefig("{}/compare_ksd_time_breakdown_LAM{}.png".format(base_dir, args.model_instances, args.l2))
		plt.clf()
		plt.close()




	elif args.procedure == 'compare_ksd_best_bw':
		ksd_mean_durations, ksd_std_durations = [], []
		ksd_mean_pows, ksd_std_pows = [], []
		ksd_mean_boot_durations, ksd_mean_stat_durations = [], []
		ksd_std_boot_durations, ksd_std_stat_durations = [], []

		neural_mean_durations, neural_std_durations = [], []
		neural_mean_pows, neural_std_pows = [], []
		neural_train_durations, neural_boot_durations, neural_stat_durations = [], [], []
		neural_train_durations_std, neural_boot_durations_std, neural_stat_durations_std = [], [], []
		neural_valid_durations = []

		ksd_dir = '{}/KSD_results/{}runs_'.format(base_dir, args.n_runs)
		if args.stage_lambda:
			neural_dir = '{}/critics_{}replicas_STAGELAM{:.1f}_beta{}'.format(base_dir, args.model_instances, args.lam_init, args.beta)
		else:
			neural_dir = '{}/critics_{}replicas_LAM{}'.format(base_dir, args.model_instances, args.l2)

		sample_sizes = [100, 200, 300, 500, 1000, 1500, 2500]
		if args.dim == 50 and args.stage_lambda:
			selected_epochs = [40, 35, 40, 40, 40, 40, 40]

		for i in range(len(sample_sizes)):

			if args.bw_factor == 1.:
				ksd_mean_durations.append(np.load("{}{}boot_{}test/mean_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
				ksd_std_durations.append(np.load("{}{}boot_{}test/std_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
				
				ksd_mean_boot_durations.append(np.load("{}{}boot_{}test/mean_boot_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
				ksd_mean_stat_durations.append(np.load("{}{}boot_{}test/mean_test_stat_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
				
				ksd_std_boot_durations.append(np.load("{}{}boot_{}test/std_boot_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
				ksd_std_stat_durations.append(np.load("{}{}boot_{}test/std_test_stat_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
				
				ksd_mean_pows.append(np.load("{}{}boot_{}test/mean_power.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))
				ksd_std_pows.append(np.load("{}{}boot_{}test/std_power.npy".format(ksd_dir, args.n_boot, sample_sizes[i])))

			else:
				ksd_mean_durations.append(np.load("{}{}boot_{}test_{}bw/mean_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i], args.bw_factor)))
				ksd_std_durations.append(np.load("{}{}boot_{}test_{}bw/std_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i], args.bw_factor)))
				
				ksd_mean_boot_durations.append(np.load("{}{}boot_{}test_{}bw/mean_boot_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i], args.bw_factor)))
				ksd_mean_stat_durations.append(np.load("{}{}boot_{}test_{}bw/mean_test_stat_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i], args.bw_factor)))
				
				ksd_std_boot_durations.append(np.load("{}{}boot_{}test_{}bw/std_boot_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i], args.bw_factor)))
				ksd_std_stat_durations.append(np.load("{}{}boot_{}test_{}bw/std_test_stat_duration.npy".format(ksd_dir, args.n_boot, sample_sizes[i], args.bw_factor)))
				
				ksd_mean_pows.append(np.load("{}{}boot_{}test_{}bw/mean_power.npy".format(ksd_dir, args.n_boot, sample_sizes[i], args.bw_factor)))
				ksd_std_pows.append(np.load("{}{}boot_{}test_{}bw/std_power.npy".format(ksd_dir, args.n_boot, sample_sizes[i], args.bw_factor)))
			

			neural_sample_dir = '{}_{}sample'.format(neural_dir, sample_sizes[i])
			
			neural_mean_durations.append(np.load("{}/mean_duration_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))
			neural_std_durations.append(np.load("{}/std_duration_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))

			neural_train_durations.append(np.load("{}/mean_train_duration.npy".format(neural_sample_dir)))
			neural_valid_durations.append(np.load("{}/valid_time.npy".format(neural_sample_dir)))
			neural_boot_durations.append(np.load("{}/mean_boot_duration_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))
			neural_stat_durations.append(np.load("{}/mean_test_stat_duration_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))
			
			neural_train_durations_std.append(np.load("{}/std_train_duration.npy".format(neural_sample_dir)))
			neural_boot_durations_std.append(np.load("{}/std_boot_duration_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))
			neural_stat_durations_std.append(np.load("{}/std_test_stat_duration_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))
			
			neural_mean_pows.append(np.load("{}/mean_power_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))
			neural_std_pows.append(np.load("{}/std_power_epoch{}_{}runs_{}boot.npy".format(neural_sample_dir, selected_epochs[i], args.n_runs, args.n_boot)))


		plt.figure(figsize=(9,7), facecolor='white', dpi=300)

		plt.errorbar(sample_sizes, neural_mean_pows, yerr=neural_std_pows, c='k', marker="x")
		plt.plot(sample_sizes, neural_mean_pows, c='k', label='Neural')

		plt.errorbar(sample_sizes, ksd_mean_pows, yerr=ksd_std_pows, c='r', marker="x")
		plt.plot(sample_sizes, ksd_mean_pows, c='r', label='KSD')

		plt.title('Power Comparison',fontsize=28)
		plt.xlabel("Sample Size",fontsize=28)
		plt.ylabel("Testing Power",fontsize=28)
		plt.xticks(fontsize=24)
		plt.yticks(fontsize=24)
		plt.grid()
		plt.legend(fontsize=24)
		plt.tight_layout()
		if args.stage_lambda:
			plt.savefig("{}/compare_ksd_{}bw_power_STAGELAM.png".format(base_dir, args.bw_factor))
		else:
			plt.savefig("{}/compare_ksd_{}bw_power_LAM{}.png".format(base_dir, args.bw_factor, args.l2))
		plt.clf()
		plt.close()


		plt.figure(figsize=(9,7), facecolor='white', dpi=300)

		neural_whole_time_mean = np.array(neural_train_durations)+np.array(neural_stat_durations)+np.array(neural_boot_durations)+np.array(neural_valid_durations)
		neural_whole_time_std = np.power(np.power(np.array(neural_train_durations_std), 2) + np.power(np.array(neural_stat_durations_std), 2) + np.power(np.array(neural_boot_durations_std), 2), 0.5)

		plt.errorbar(sample_sizes, neural_whole_time_mean, yerr=neural_whole_time_std, c='k', marker="x")
		plt.plot(sample_sizes, neural_whole_time_mean, c='k', label='Neural')

		ksd_whole_time_mean = np.array(ksd_mean_stat_durations)+np.array(ksd_mean_boot_durations)
		ksd_whole_time_std = np.power(np.power(np.array(ksd_std_stat_durations), 2) + np.power(np.array(ksd_std_boot_durations), 2), 0.5)

		plt.errorbar(sample_sizes, ksd_whole_time_mean, yerr=ksd_whole_time_std, c='r', marker="x")
		plt.plot(sample_sizes, ksd_whole_time_mean, c='r', label='KSD')

		plt.title('Total Time Comparison',fontsize=28)
		plt.xlabel("Sample Size",fontsize=28)
		plt.ylabel("Time (s)",fontsize=28)
		plt.xticks(fontsize=24)
		plt.yticks(fontsize=24)
		plt.grid()
		plt.legend(fontsize=24)
		plt.tight_layout()
		if args.stage_lambda:
			plt.savefig("{}/compare_ksd_{}bw_time_STAGELAM.png".format(base_dir, args.bw_factor))
		else:
			plt.savefig("{}/compare_ksd_{}bw_time_LAM{}.png".format(base_dir, args.bw_factor, args.l2))
		plt.clf()
		plt.close()


		plt.figure(figsize=(9,7), facecolor='white', dpi=300)

		neural_whole_test_stat_mean = np.array(neural_train_durations)+np.array(neural_stat_durations)+np.array(neural_valid_durations)
		neural_whole_test_stat_std = np.power(np.power(np.array(neural_train_durations_std), 2) + np.power(np.array(neural_stat_durations_std), 2), 0.5)

		plt.errorbar(sample_sizes, neural_whole_test_stat_mean, yerr=neural_whole_test_stat_std, c='k', ls='-.', marker="x")
		plt.plot(sample_sizes, neural_whole_test_stat_mean, ls='-.', c='k', label='Neural Test Statistic')

		plt.errorbar(sample_sizes, neural_boot_durations, yerr=neural_boot_durations_std, c='k', ls='--', marker="x")
		plt.plot(sample_sizes, neural_boot_durations, c='k', ls='--', label=r'Neural Bootstrap $(n=500)$')

		plt.errorbar(sample_sizes, ksd_mean_stat_durations, yerr=ksd_std_stat_durations, c='r', ls='-.', marker="x")
		plt.plot(sample_sizes, ksd_mean_stat_durations, c='r', ls='-.', label='KSD Test Statistic')

		plt.errorbar(sample_sizes, ksd_mean_boot_durations, yerr=ksd_std_boot_durations, c='r', ls='--', marker="x")
		plt.plot(sample_sizes, ksd_mean_boot_durations, c='r', ls='--', label=r'KSD Bootstrap $(n=500)$')

		plt.title('Breakdown Time Comparison',fontsize=28)
		plt.xlabel("Sample Size",fontsize=28)
		plt.ylabel("Time (s)",fontsize=28)
		plt.xticks(fontsize=24)
		plt.yticks(fontsize=24)
		plt.grid()
		plt.legend(fontsize=24)
		plt.tight_layout()
		if args.stage_lambda:
			plt.savefig("{}/compare_ksd_{}bw_time_breakdown_STAGELAM.png".format(base_dir, args.bw_factor))
		else:
			plt.savefig("{}/compare_ksd_{}bw_time_breakdown_LAM{}.png".format(base_dir, args.bw_factor, args.l2))
		plt.clf()
		plt.close()




	elif args.procedure == 'compare_ksd_bw':
		if args.stage_lambda:
			neural_dir = '{}/critics_{}replicas_STAGELAM{:.1f}_beta{}_{}sample'.format(base_dir, args.model_instances, args.lam_init, args.beta, args.n_test)
		else:
			neural_dir = '{}/critics_{}replicas_LAM{}_{}sample'.format(base_dir, args.model_instances, args.l2, args.n_test)

		if args.n_test == 500:
			selected_epoch = 40
		elif args.n_test == 750:
			selected_epoch = 30
		elif args.n_test == 800:
			selected_epoch = 35
		elif args.n_test == 1000:
			selected_epoch = 40
		elif args.n_test == 1500:
			selected_epoch = 40

		neural_mean_pow = np.load("{}/mean_power_epoch{}_{}runs_{}boot.npy".format(neural_dir, selected_epoch, args.n_runs, args.n_boot))
		neural_std_pow = np.load("{}/std_power_epoch{}_{}runs_{}boot.npy".format(neural_dir, selected_epoch, args.n_runs, args.n_boot))
		
		bws = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
		ksd_mean_pows, ksd_std_pows = [], []

		ksd_dir = '{}/KSD_results/{}runs_{}boot_{}test'.format(base_dir, args.n_runs, args.n_boot, args.n_test)
		for i in range(len(bws)):
			if bws[i] == 1.:
				ksd_mean_pows.append(np.load("{}/mean_power.npy".format(ksd_dir)))
				ksd_std_pows.append(np.load("{}/std_power.npy".format(ksd_dir)))
			else:
				ksd_mean_pows.append(np.load("{}_{}bw/mean_power.npy".format(ksd_dir, bws[i])))
				ksd_std_pows.append(np.load("{}_{}bw/std_power.npy".format(ksd_dir, bws[i])))

		plt.figure(figsize=(9,7), facecolor='white', dpi=300)

		plt.errorbar(bws, ksd_mean_pows, yerr=ksd_std_pows, c='r', marker="x")
		plt.plot(bws, ksd_mean_pows, c='r', label='KSD')

		plt.axhline(neural_mean_pow, c='k', label='Neural')
		plt.axhspan(neural_mean_pow-neural_std_pow, neural_mean_pow+neural_std_pow, color='k', alpha=0.1)
		
		plt.ylim(-0.05, 1.05)
		plt.xscale('log')

		plt.title('Bandwidth Power Comparison',fontsize=28)
		plt.xlabel("Bandwidth Multiplicative Factor",fontsize=28)
		plt.ylabel("Testing Power",fontsize=28)
		plt.xticks(fontsize=24)
		plt.yticks(fontsize=24)
		plt.grid()
		plt.legend(fontsize=24)
		plt.tight_layout()
		if args.stage_lambda:
			plt.savefig("{}/compare_ksd_bw_STAGELAM_{}test.png".format(base_dir, args.n_test))
		else:
			plt.savefig("{}/compare_ksd_bw_LAM{}_{}test.png".format(base_dir, args.l2, args.n_test))
		plt.clf()
		plt.close()




	elif args.procedure == 'compare_ksd_bw_multiple':
		sample_sizes = [500, 750, 800, 1000, 1500]
		selected_epochs = [40, 30, 35, 35, 25]
		bws = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
		
		fig, ax = plt.subplots(1, len(sample_sizes), figsize=(9*len(sample_sizes),7), facecolor='white', dpi=300)
		for k in range(len(sample_sizes)):
			if args.stage_lambda:
				neural_dir = '{}/critics_{}replicas_STAGELAM{:.1f}_beta{}_{}sample'.format(base_dir, args.model_instances, args.lam_init, args.beta, sample_sizes[k])
			else:
				neural_dir = '{}/critics_{}replicas_LAM{}_{}sample'.format(base_dir, args.model_instances, args.l2, sample_sizes[k])

			selected_epoch = selected_epochs[k]

			neural_mean_pow = np.load("{}/mean_power_epoch{}_{}runs_{}boot.npy".format(neural_dir, selected_epoch, args.n_runs, args.n_boot))
			neural_std_pow = np.load("{}/std_power_epoch{}_{}runs_{}boot.npy".format(neural_dir, selected_epoch, args.n_runs, args.n_boot))
			
			ksd_mean_pows, ksd_std_pows = [], []

			ksd_dir = '{}/KSD_results/{}runs_{}boot_{}test'.format(base_dir, args.n_runs, args.n_boot, sample_sizes[k])
			for i in range(len(bws)):
				if bws[i] == 1.:
					ksd_mean_pows.append(np.load("{}/mean_power.npy".format(ksd_dir)))
					ksd_std_pows.append(np.load("{}/std_power.npy".format(ksd_dir)))
				else:
					ksd_mean_pows.append(np.load("{}_{}bw/mean_power.npy".format(ksd_dir, bws[i])))
					ksd_std_pows.append(np.load("{}_{}bw/std_power.npy".format(ksd_dir, bws[i])))


			ax[k].errorbar(bws, ksd_mean_pows, yerr=ksd_std_pows, c='k', marker="x")
			ax[k].plot(bws, ksd_mean_pows, c='k')

			ax[k].axhline(neural_mean_pow, c='r', label='Neural Critic Power')
			ax[k].axhspan(neural_mean_pow-neural_std_pow, neural_mean_pow+neural_std_pow, color='r', alpha=0.1)
			
			ax[k].set_ylim(-0.05, 1.05)
			ax[k].set_xscale('log')

			ax[k].set_title('Sample Size: {}'.format(sample_sizes[k]),fontsize=28)
			ax[k].set_xlabel("Bandwidth Multiplicative Factor",fontsize=28)
			ax[k].set_ylabel("Testing Power",fontsize=28)
			ax[k].tick_params(axis='both',labelsize=24)
			ax[k].grid()
			ax[k].legend(fontsize=24)
		plt.tight_layout()
		if args.stage_lambda:
			plt.savefig("{}/compare_ksd_bw_multiple_STAGELAM.png".format(base_dir))
		else:
			plt.savefig("{}/compare_ksd_bw_multiple_LAM{}.png".format(base_dir, args.l2))
		plt.clf()
		plt.close()