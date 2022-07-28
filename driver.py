import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from time import sleep
import os
import arrow
import utils.log_utils as log_utils
from utils.class_utils import LambdaStager
from regularized_training import RegularizedTrainer
from mnist_training import MNISTRegularizedTrainer
from mnist_analysis import MNISTAnalyzer
from hypothesis_testing import HypothesisTester

parser = argparse.ArgumentParser()


# main selection
parser.add_argument('--procedure', type=str, default='train_replicas', choices=['fit', 'train_replicas_plot', 'train_replicas', 'mnist_examine', 'ksd_power', 'neural_power', 'compare'])


# model parameters
parser.add_argument('--dim', type=int, default=1)
parser.add_argument('--mean_pert', type=float, default=1.)
parser.add_argument('--std_pert', type=float, default=1.)
parser.add_argument('--weight_pert', type=float, default=1.)


# MNIST parameters
parser.add_argument('--mnist', action='store_true')
parser.add_argument('--mix_prop', type=float, default=0.97)


# network and penalization paramters
parser.add_argument('--activation', type=str, default='swish', choices=['swish','softplus'])
parser.add_argument('--nn_layers', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--n_epochs', type=int, default=35)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optim', type=str, default='adam', choices=['adam','sgd'])

parser.add_argument('--n_train', type=int, default=80)
parser.add_argument('--n_val', type=int, default=20)
parser.add_argument('--n_test', type=int, default=100)


# regularization and staging parameters
parser.add_argument('--l2', type=float, default=0.01)
parser.add_argument('--l2_lower', type=int, default=-2)
parser.add_argument('--l2_upper', type=int, default=2)

parser.add_argument('--stage_lambda', action='store_true')
parser.add_argument('--lam_init', type=float, default=5e-1)
parser.add_argument('--beta', type=float, default=0.85)
parser.add_argument('--lam_term', type=float, default=5e-3)


# hypothesis testing parameters
parser.add_argument('--n_runs', type=int, default=1)
parser.add_argument('--n_boot', type=int, default=500)
parser.add_argument('--alpha', type=float, default=.05)

parser.add_argument('--bw_factor', type=float, default=1.)


# model loading parameters
parser.add_argument('--model_instances', type=int, default=20)
parser.add_argument('--model_index', type=int, default=0)
parser.add_argument('--epoch_checkpoint', type=int, default=1)
parser.add_argument('--stop_epoch', type=int, default=45)


# logger parameters
parser.add_argument('--report_freq', type=int, default=5)
parser.add_argument('--mse_sample', type=int, default=20000)


parser.set_defaults(feature=False)


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


if __name__ == "__main__":
	args = parser.parse_args()

	n_sample = args.n_train + args.n_val + args.n_test

	if not os.path.exists("results"):
		os.mkdir("results")

	if args.mnist:
		if not os.path.exists("results/mnist"):
			os.mkdir("results/mnist")

		base_dir = "results/mnist/prop{}".format(args.mix_prop)

	else:
		if not os.path.exists("results/dim_{}".format(args.dim)):
			os.mkdir("results/dim_{}".format(args.dim))

		if args.dim <= 1:
			base_dir = "results/dim_{}/mean{}_std{}".format(args.dim, args.mean_pert, args.std_pert)
		else:
			base_dir = "results/dim_{}/weight{}_std{}".format(args.dim, args.weight_pert, args.std_pert)

	if not os.path.exists(base_dir):
		os.mkdir(base_dir)




	if args.procedure == 'fit':
		if args.stage_lambda:
			lam = LambdaStager(args.lam_init, args.beta, args.lam_term)
			savedir = "{}/fit_STAGELAM{:.1f}_beta{}_{}sample".format(base_dir, args.lam_init, args.beta, n_sample)
		else:
			lam = args.l2
			savedir = "{}/fit_LAM{}_{}sample".format(base_dir, args.l2, n_sample)

		if not os.path.exists(savedir):
			os.mkdir(savedir)

		if args.mnist:
			trainer = MNISTRegularizedTrainer(args, device, l2=lam)
		else:
			trainer = RegularizedTrainer(args, device, l2=lam)

		trainer.fit_critic(savedir=savedir)

		log_utils.logger(args, "{}/specifications.txt".format(savedir))




	elif args.procedure == 'train_replicas_plot':
		if args.stage_lambda:
			lam = LambdaStager(args.lam_init, args.beta, args.lam_term)
			savedir = "{}/critics_{}replicas_STAGELAM{:.1f}_beta{}_{}sample".format(base_dir, args.model_instances, args.lam_init, args.beta, n_sample)
		else:
			lam = args.l2
			savedir = "{}/critics_{}replicas_LAM{}_{}sample".format(base_dir, args.model_instances, lam, n_sample)

		if not os.path.exists(savedir):
			os.mkdir(savedir)

		checkpoints = [20, 25, 30, 35, 40, 45, 50, 55, 60]

		for i in range(args.model_instances):
			print("\nTraining Replica {}".format(i+1))
			sleep(np.random.uniform(0, 0.1))
			
			start_time = arrow.now()

			if not os.path.exists("{}/critic_{}".format(savedir, i)):
				os.mkdir("{}/critic_{}".format(savedir, i))
			else:
				continue

			if args.stage_lambda:
				lam = LambdaStager(args.lam_init, args.beta, args.lam_term)

			if args.mnist:
				trainer = MNISTRegularizedTrainer(args, device, l2=lam)
				validation_stein_discs, validation_mse, test_sd_ratio, train_stein_discs = trainer.fit_critic(plot=False,
					savedir="{}/critic_{}".format(savedir, i), checkpoints=checkpoints)
			else:
				trainer = RegularizedTrainer(args, device, l2=lam)
				mse, batch_mse, validation_stein_discs, validation_mse, save_critic_times, epoch_times, batch_times = trainer.fit_critic(plot=False,
					savedir="{}/critic_{}".format(savedir, i), checkpoints=checkpoints)

			np.save("{}/critic_{}/validation_mse.npy".format(savedir, i), np.array(validation_mse))
			np.save("{}/critic_{}/training_duration.npy".format(savedir, i), np.array((arrow.now()-start_time).total_seconds()))
			
			if args.stage_lambda:
				np.save("{}/critic_{}/staged_lams.npy".format(savedir, i), np.array(lam.lams))
				plt.figure(figsize=(9,7))
				plt.plot(lam.lams)
				plt.yscale('log')
				plt.grid()
				plt.savefig("{}/critic_{}/staged_lams.png".format(savedir, i))
			
			if not args.mnist:
				np.save("{}/critic_{}/mse.npy".format(savedir, i), np.array(mse))
				np.save("{}/critic_{}/train_mse.npy".format(savedir, i), np.array(batch_mse))
				np.save("{}/critic_{}/save_critic_duration.npy".format(savedir, i), np.sum(save_critic_times))
				np.save("{}/critic_{}/batch_times.npy".format(savedir, i), batch_times)
				np.save("{}/critic_{}/epoch_times.npy".format(savedir, i), epoch_times)
			else:
				np.save("{}/critic_{}/test_sd_ratio.npy".format(savedir, i), np.array(test_sd_ratio))
				np.save("{}/critic_{}/train_stein_discs.npy".format(savedir, i), np.array(train_stein_discs))

			np.save("{}/critic_{}/validation_stein_discs.npy".format(savedir, i), np.array(validation_stein_discs))
			log_utils.logger(args, "{}/critic_{}/specifications.txt".format(savedir, i))




	elif args.procedure == 'train_replicas':
		if args.stage_lambda:
			lam = LambdaStager(args.lam_init, args.beta, args.lam_term)
			savedir = "{}/critics_{}replicas_STAGELAM{:.1f}_beta{}_{}sample".format(base_dir, args.model_instances, args.lam_init, args.beta, n_sample)
		else:
			lam = args.l2
			savedir = "{}/critics_{}replicas_LAM{}_{}sample".format(base_dir, args.model_instances, lam, n_sample)

		if not os.path.exists(savedir):
			os.mkdir(savedir)

		if args.mnist:
			assert False
		
		if args.stage_lambda:
			checkpoints = [20, 25, 30, 35, 40, 45, 50, 55, 60]
		else:
			checkpoints = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60]

		for i in range(args.model_instances):
			
			start_time = arrow.now()

			if not os.path.exists("{}/critic_{}".format(savedir, i)):
				os.mkdir("{}/critic_{}".format(savedir, i))
			else:
				continue
			
			if args.stage_lambda:
				lam = LambdaStager(args.lam_init, args.beta, args.lam_term)
			
			trainer = RegularizedTrainer(args, device, l2=lam)
			validation_mse, save_critic_times, form_batch_times, compute_valid_times, epoch_times, batch_times = trainer.fit_critic_efficient(savedir="{}/critic_{}".format(savedir, i),
				checkpoints=checkpoints)

			np.save("{}/critic_{}/validation_mse.npy".format(savedir, i), np.array(validation_mse))
			np.save("{}/critic_{}/training_duration.npy".format(savedir, i), np.array((arrow.now()-start_time).total_seconds()))
			
			if args.stage_lambda:
				np.save("{}/critic_{}/staged_lams.npy".format(savedir, i), np.array(lam.lams))
				plt.figure(figsize=(9,7))
				plt.plot(lam.lams)
				plt.yscale('log')
				plt.grid()
				plt.savefig("{}/critic_{}/staged_lams.png".format(savedir, i))
			
			np.save("{}/critic_{}/save_critic_duration.npy".format(savedir, i), np.sum(save_critic_times))
			
			np.save("{}/critic_{}/form_batch_times.npy".format(savedir, i), form_batch_times)
			np.save("{}/critic_{}/compute_valid_times.npy".format(savedir, i), compute_valid_times)

			np.save("{}/critic_{}/batch_times.npy".format(savedir, i), batch_times)
			np.save("{}/critic_{}/epoch_times.npy".format(savedir, i), epoch_times)

			log_utils.logger(args, "{}/critic_{}/specifications.txt".format(savedir, i))




	elif args.procedure == 'mnist_examine':
		if args.stage_lambda:
			savedir = "{}/critics_{}replicas_STAGELAM{:.1f}_beta{}_{}sample".format(base_dir, args.model_instances, args.lam_init, args.beta, n_sample)
			lam = np.load("{}/critic_{}/staged_lams.npy".format(savedir, args.model_index))[args.epoch_checkpoint - 1]
		else:
			savedir = "{}/critics_{}replicas_LAM{}_{}sample".format(base_dir, args.model_instances, args.l2, n_sample)
			lam = args.l2

		if not args.mnist:
			# this is only for mnist experiments
			assert False

		ckpt = "{}/critic_{}/critic_epoch_{}".format(savedir, args.model_index, args.epoch_checkpoint)
		print(ckpt)
		assert os.path.exists(ckpt)

		analyzer = MNISTAnalyzer(args, device, lam)
		analyzer.analyze_critic(ckpt, savedir+"/critic_{}".format(args.model_index))




	elif args.procedure == 'ksd_power':
		if args.bw_factor == 1.:
			base_savedir = "{}/KSD_results/{}runs_{}boot_{}test".format(base_dir, args.n_runs, args.n_boot, args.n_test)
		else:
			base_savedir = "{}/KSD_results/{}runs_{}boot_{}test_{}bw".format(base_dir, args.n_runs, args.n_boot, args.n_test, args.bw_factor)

		if args.mnist:
			# this is only for non-mnist experiments
			assert False

		if not os.path.exists("{}/KSD_results".format(base_dir)):
			os.mkdir("{}/KSD_results".format(base_dir))
		
		if not os.path.exists(base_savedir):
			os.mkdir(base_savedir)

		sleep(np.random.uniform(0, 0.1))
		counter = 0
		while os.path.exists(base_savedir+"/sample_{}".format(counter)):
			counter += 1
		os.mkdir(base_savedir+"/sample_{}".format(counter))
		savedir = base_savedir+"/sample_{}".format(counter)

		tester = HypothesisTester(args, device)
		power, mean_test_stat_duration, mean_boot_duration = tester.get_power()
		
		np.save(savedir+"/power.npy", np.array([power]))
		np.save(savedir+"/test_stat_duration.npy", np.array([mean_test_stat_duration]))
		np.save(savedir+"/boot_duration.npy", np.array([mean_boot_duration]))
		
		log_utils.power_logger(args, power, "{}/power_report.txt".format(savedir))




	elif args.procedure == 'neural_power':
		if args.stage_lambda:
			savedir = "{}/critics_{}replicas_STAGELAM{:.1f}_beta{}_{}sample".format(base_dir, args.model_instances, args.lam_init, args.beta, n_sample)
		else:
			savedir = "{}/critics_{}replicas_LAM{}_{}sample".format(base_dir, args.model_instances, args.l2, n_sample)

		if args.mnist:
			# this is only for non-mnist experiments
			assert False
		
		e = args.stop_epoch
		critic_ids = torch.arange(args.model_instances)
		for critic_id in critic_ids:
			critic_dir = "{}/critic_{}".format(savedir, critic_id)

			print("\nCritic {} Epoch {}".format(critic_id, e))
			this_power_dir = "{}/power_epoch{:03d}_{}runs_{}boot.npy".format(critic_dir, e, args.n_runs, args.n_boot)
			this_test_stat_duration_dir = "{}/test_stat_duration_epoch{:03d}_{}runs_{}boot.npy".format(critic_dir, e, args.n_runs, args.n_boot)
			this_boot_duration_dir = "{}/boot_duration_epoch{:03d}_{}runs_{}boot.npy".format(critic_dir, e, args.n_runs, args.n_boot)
			
			ckpt = "{}/critic_epoch_{}".format(critic_dir, e)
			
			tester = HypothesisTester(args, device)
			power, mean_test_stat_duration, mean_boot_duration = tester.get_power(ckpt)
			np.save(this_test_stat_duration_dir, np.array([mean_test_stat_duration]))
			np.save(this_boot_duration_dir, np.array([mean_boot_duration]))
			np.save(this_power_dir, np.array([power]))
			log_utils.power_logger(args, power, "{}/power_report_epoch{:03d}.txt".format(critic_dir, e))
	



	elif args.procedure == 'compare':
		lams = list(np.logspace(args.l2_lower, args.l2_upper, args.l2_upper-args.l2_lower+1))
		if args.stage_lambda:
			stager = LambdaStager(args.lam_init, args.beta, args.lam_term)
			lams = lams + [stager]
			savedir = "{}/compare_STAGELAM{}-{}".format(base_dir, lams[0], lams[-2])
		else:
			savedir = "{}/compare_LAM{}-{}".format(base_dir, lams[0], lams[-1])

		if not os.path.exists(savedir):
			os.mkdir(savedir)

		if args.mnist:
			dummy_trainer = MNISTRegularizedTrainer(args, device, l2=1.)
			data_val = dummy_trainer.data_test
		else:
			dummy_trainer = RegularizedTrainer(args, device, l2=1.)
			data_val = dummy_trainer.data_val

		data_train = dummy_trainer.data_train

		lam_stein_discs, lam_validation_stein_discs, lam_losses, lam_penalties, lam_mse, lam_batch_mse = [], [], [], [], [], []
		for lam in lams:
			print("\nTraining Lambda {}".format('Staged' if type(lam) == LambdaStager else lam))

			if args.mnist:
				trainer = MNISTRegularizedTrainer(args, device, l2=lam)
				trainer.data_test = data_val.requires_grad_().to(device)
			else:
				trainer = RegularizedTrainer(args, device, l2=lam)
				trainer.data_val = data_val.requires_grad_().to(device)

			trainer.data_train = data_train.to(device)
			
			stein_discs, validation_stein_discs, losses, penalties, mse, batch_mse = trainer.fit_critic(plot=False, compare_dir=savedir)

			factor = 1 if args.stage_lambda else lam

			lam_stein_discs.append(factor * torch.tensor(stein_discs).detach().numpy())
			lam_validation_stein_discs.append(factor * torch.tensor(validation_stein_discs).detach().numpy())
			lam_losses.append(factor * torch.tensor(losses).detach().numpy())
			lam_penalties.append(factor * torch.tensor(penalties).detach().numpy())
			if args.mnist:
				lam_mse = None
				lam_batch_mse = None
			else:
				lam_mse.append(torch.tensor(mse).detach().numpy())
				lam_batch_mse.append(torch.tensor(batch_mse).detach().numpy())

		log_utils.result_plotter(lams, lam_stein_discs, lam_validation_stein_discs, lam_losses, lam_penalties, lam_mse, lam_batch_mse, savedir)
		log_utils.logger(args, "{}/specifications.txt".format(savedir))