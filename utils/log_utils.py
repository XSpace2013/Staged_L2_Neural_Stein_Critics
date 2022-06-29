import matplotlib.pyplot as plt
import numpy as np
from .class_utils import LambdaStager


color_list = ['gold', 'tab:cyan', 'tab:pink','tab:blue','tab:orange','tab:green','tab:red','tab:purple', 'chartreuse']
# color_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:pink']

replica_color_list = ['lightcoral', 'red', 'peru', 'darkorange', 'gold', 'yellow', 'darkolivegreen', 'chartreuse', 'forestgreen', 'aquamarine',
		'darkslategray', 'darkturquoise', 'deepskyblue', 'mediumpurple', 'indigo', 'mediumorchid', 'violet', 'deeppink', 'lightpink', 'k']


def batch_plotter(lams, field, window_size, fieldname, savedir):
	fig = plt.figure(figsize=(9,7))
	for i in range(len(lams)):
		plt.plot(list(range(1, window_size+1)), field[i][:window_size],
			color=color_list[i] if len(lams) > 1 else 'k', ls='--', alpha=0.5)
		plt.plot(list(range(window_size+1, len(field[i])+1)), field[i][window_size:],
			color=color_list[i] if len(lams) > 1 else 'k',
			label="Staging" if type(lams[i]) == LambdaStager else r"$\lambda: $" + str(lams[i]))
	plt.ylabel("{} (Window={})".format(fieldname, window_size) if window_size != 0 else fieldname,fontsize=14)
	plt.xlabel("Training Batches",fontsize=14)
	if fieldname == "Training MSE":
		plt.yscale('linear')
	else:
		plt.yscale('symlog')
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	if len(lams) > 1:
		plt.legend(fontsize=14)
	plt.tight_layout()
	plt.savefig(savedir)
	plt.clf()
	plt.close()


def epoch_plotter(lams, field, fieldname, savedir):
	fig = plt.figure(figsize=(9,7))
	for i in range(len(lams)):
		plt.plot(field[i],
			color=color_list[i] if len(lams) > 1 else 'k',
			label="Staging" if type(lams[i]) == LambdaStager else r"$\lambda: $" + str(lams[i]))
	plt.ylabel(fieldname, fontsize=14)
	plt.xlabel("Training Epochs", fontsize=14)
	if fieldname == "MSE" or fieldname == "Validation MSE":
		plt.yscale('linear')
	else:
		plt.yscale('symlog')
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.legend(fontsize=14)
	plt.tight_layout()
	plt.savefig(savedir)
	plt.clf()
	plt.close()


def epoch_plotter_twofields(lams, field1, field2, fieldname1, fieldname2, savedir):
	fig = plt.figure(figsize=(9,7))
	plt.plot(field[i],
		color=color_list[i] if len(lams) > 1 else 'k',
		label="Staging" if type(lams[i]) == LambdaStager else r"$\lambda: $" + str(lams[i]))
	plt.ylabel(fieldname, fontsize=14)
	plt.xlabel("Training Epochs", fontsize=14)
	if fieldname == "MSE" or fieldname == "Validation MSE":
		plt.yscale('linear')
	else:
		plt.yscale('symlog')
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.legend(fontsize=14)
	plt.tight_layout()
	plt.savefig(savedir)
	plt.clf()
	plt.close()


def epoch_to_batches_plotter(field, fieldname, savedir, batches_per_epoch):
	field_array = np.array(field)
	field_mean = field_array.mean(axis=0)
	field_std = field_array.std(axis=0)
	
	fig = plt.figure(figsize=(9,7))
	if fieldname == "MSE":
		batch_range = np.array(list(range(len(field_mean)))) * batches_per_epoch
	else:
		batch_range = np.array(list(range(1,len(field_mean)+1))) * batches_per_epoch
	if len(field) == 1:
		plt.plot(batch_range, field[0], color='k')
	else:
		plt.plot(batch_range, field_mean, color='k', label="Mean")
		plt.fill_between(batch_range, field_mean-field_std, field_mean+field_std,
			color='r', label="Standard Deviation", alpha=0.25)
		plt.axhline(0, ls='--', color='k', alpha=0.25, label='SD: 0')
		plt.legend(fontsize=14)
	plt.ylabel(fieldname, fontsize=14)
	plt.xlabel("Training Batches", fontsize=14)
	if fieldname == "MSE" or fieldname == "Validation MSE":
		plt.yscale('linear')
	else:
		plt.yscale('symlog')
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.tight_layout()
	plt.savefig(savedir)
	plt.clf()
	plt.close()


def ma_stability_plotter(field_mean, field_std, window_size, fieldname, savedir):
	fig = plt.figure(figsize=(9,7))
	plt.plot(list(range(1, window_size+1)), field_mean[:window_size],
			color='k', ls='--', alpha=0.5)
	plt.plot(list(range(window_size+1, len(field_mean)+1)), field_mean[window_size:], color='k', label="Mean")
	plt.fill_between(list(range(window_size+1, len(field_mean)+1)),
		field_mean[window_size:]-field_std[window_size:], field_mean[window_size:]+field_std[window_size:],
		color='r', label="Standard Deviation", alpha=0.25)
	plt.ylabel(fieldname, fontsize=14)
	plt.xlabel("Training Batches", fontsize=14)
	plt.yscale('symlog')
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.legend(fontsize=14)
	plt.tight_layout()
	plt.savefig(savedir)
	plt.clf()
	plt.close()


def batch_power_plotter(gof_epochs, power, savedir, batches_per_epoch):
	fig = plt.figure(figsize=(9,7))
	plt.scatter(np.array(gof_epochs) * batches_per_epoch, power, c='k', s=50)
	plt.ylabel("Testing Power",fontsize=14)
	plt.xlabel("Training Batches",fontsize=14)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.savefig(savedir)
	plt.clf()
	plt.close()


def batch_replica_plotter(field, fieldname, savedir, batches_per_epoch=None, window_size=None):
	assert not (window_size is not None and batches_per_epoch is not None)
	
	if batches_per_epoch is not None:
		max_epochs = max([len(replica_field) for replica_field in field])
		batch_range = np.array(list(range(0,max_epochs))) * batches_per_epoch

	fig = plt.figure(figsize=(9,7))
	
	for i in range(len(field)):
		if batches_per_epoch is not None:
			plt.plot(batch_range[:len(field[i])], field[i],
				color=replica_color_list[i % len(replica_color_list)] if len(field) > 1 else 'k',
				label="Model {}".format(i))
		else:
			plt.plot(list(range(1, len(field[i])+1)), field[i],
				color=replica_color_list[i % len(replica_color_list)] if len(field) > 1 else 'k',
				label="Model {}".format(i))

	plt.ylabel("{} (Window={})".format(fieldname, window_size) if window_size is not None else fieldname,fontsize=14)
	plt.xlabel("Training Batches",fontsize=14)
	if fieldname == "MSE":
		plt.yscale('linear')
	else:
		plt.yscale('symlog')
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.legend(fontsize=12, loc=3)
	plt.tight_layout()
	plt.savefig(savedir)
	plt.clf()
	plt.close()


def batch_replica_dist_plotter(field, fieldname, savedir, batches_per_epoch=None):
	
	max_epochs = max([len(replica_field) for replica_field in field])
	filled_replicas = []
	for replica_field in field:
		filled_replica = list(replica_field)
		while len(filled_replica) < max_epochs:
			filled_replica.append(filled_replica[0])
		filled_replicas.append(filled_replica)
	filled_replicas = np.array(filled_replicas)

	if batches_per_epoch is not None:
		batch_range = np.array(list(range(0,max_epochs))) * batches_per_epoch
	else:
		batch_range = list(range(1,max_epochs+1))

	fig = plt.figure(figsize=(9,7))

	replica_mean = filled_replicas.mean(axis=0)
	replica_std = filled_replicas.std(axis=0)

	plt.plot(batch_range, replica_mean, color='k', label='Mean')
	plt.fill_between(batch_range, replica_mean - replica_std, replica_mean + replica_std,
		color='r', label="Standard Deviation", alpha=0.25)

	plt.ylabel(fieldname,fontsize=14)
	plt.xlabel("Training Batches",fontsize=14)
	plt.yscale('linear')
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.legend(fontsize=14)
	plt.tight_layout()
	plt.savefig(savedir)
	plt.clf()
	plt.close()


def result_plotter(lams, lam_stein_discs, lam_validation_stein_discs, lam_losses, lam_penalties, lam_mse, lam_batch_mse, savedir):
	if lam_batch_mse is not None:
		batch_plotter(lams, lam_batch_mse, 0, "Training MSE", "{}/training_mse.png".format(savedir))
		epoch_plotter(lams, lam_mse, "MSE", "{}/mse_traj.png".format(savedir))

	epoch_plotter(lams, lam_stein_discs, "Training Stein Discrepancy", "{}/stein_traj.png".format(savedir))
	epoch_plotter(lams, lam_validation_stein_discs, "Validation Stein Discrepancy", "{}/validation_stein_traj.png".format(savedir))
	epoch_plotter(lams, lam_losses, "Training Loss", "{}/loss_traj.png".format(savedir))
	epoch_plotter(lams, lam_penalties, "Training Norm Regularization", "{}/regularization_traj.png".format(savedir))


def logger(args, savedir):
	report_file = open(savedir, "w")
	report_file.write("activation: {}\nnn_layers: {}\n".format(args.activation, args.nn_layers))
	report_file.write("hidden_dim: {}\nn_epochs: {}\nbatch_size: {}\nlr: {}\n".format(args.hidden_dim, args.n_epochs, args.batch_size, args.lr))
	report_file.write("n_train: {}\nn_val: {}\nn_test: {}\n".format(args.n_train, args.n_val, args.n_test))
	report_file.write("lam_init: {}\nbeta: {}\nlam_term: {}\n".format(args.lam_init, args.beta, args.lam_term))
	report_file.write("model_instances: {}\nn_runs {}\nn_boot: {}\nalpha: {}\n".format(args.model_instances, args.n_runs, args.n_boot, args.alpha))
	report_file.write("mse_sample: {}".format(args.mse_sample))
	if args.mnist:
		report_file.write("\nmix_prop: {}".format(args.mix_prop))
	report_file.close()


def power_logger(args, power, savedir):
	report_file = open(savedir, "w")
	report_file.write("n_test: {}\nn_runs: {}\nn_boot: {}\nalpha: {}\npower: {}".format(args.n_test, args.n_runs,
		args.n_boot, args.alpha, power))
	report_file.close()