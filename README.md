# Neural Stein critics with staged $L^2$-regularization

> Implementation of code needed to reproduce the experiments found in:

> M. Repasky, X. Cheng, & Y. Xie. "Neural Stein critics with staged $L^2$-regularization". (https://arxiv.org/abs/2207.03406)

<!-- ## Table of Contents
* [High-Dimensional Gaussian Mixture](#high-dimensional-gaussian-mixture)
* [Comparison to KSD](#comparison-to-ksd)
* [MNIST Experiment](#mnist-experiment)
 -->
 ## High-Dimensional Gaussian Mixture
 To reproduce the comparison of MSE and power from Section 4.1, first run the following command to train the networks:
 
 ### Fixed Regularization Training
 `python driver.py --procedure 'train_replicas_plot' --dim 25 --weight_pert 0.8 --std_pert 0.5 --n_epochs 60 
 --batch_size 200 --lr 1e-3 --model_instances 10 --n_train 2000 --n_val 1000 --n_test 500 --l2 1e-3`
 
  ### Staged Regularization Training
 `python driver.py --procedure 'train_replicas_plot' --dim 25 --weight_pert 0.8 --std_pert 0.5 --n_epochs 60 
 --batch_size 200 --lr 1e-3 --model_instances 10 --n_train 2000 --n_val 1000 --n_test 500 --stage_lambda --lam_init 0.4 --beta 0.85 --lam_term 5e-4`
 
 Then, run the following sequence of commands to obtain the training epoch selected by validation, the power at that epoch, and the average power of the 10 replicas.
 
 `python plotter.py --procedure 'validation_epoch' --dim 25 --weight_pert 0.8 --std_pert 0.5 --n_epochs 60 --batch_size 200 --lr 1e-3 --model_instances 5
 --n_train 2000 --n_val 1000 --n_test 500 --l2 1e-3`
 
 `python driver.py --procedure 'neural_power' --dim 25 --weight_pert 0.8 --std_pert 0.5 --model_instances 10
 --n_train 2000 --n_val 1000 --n_test 500 --n_runs 500 --n_boot 500 --l2 1e-3 --stop_epoch STOP_EPOCH`
 
 `python plotter.py --procedure 'neural_power' --dim 25 --weight_pert 0.8 --std_pert 0.5 --model_instances 10
 --n_train 2000 --n_val 1000 --n_test 500 --n_runs 500 --n_boot 500 --l2 1e-3 --stop_epoch STOP_EPOCH`
 
 Finally, run the following two commands to plot the MSE and power comparison between the fixed and staged regularization strategies.
 
 `python plotter.py --procedure 'compare_metrics' --dim 25 --weight_pert 0.8 --std_pert 0.5 --model_instances 10
 --n_train 2000 --n_val 1000 --n_test 500 --stage_lambda`
 
  `python plotter.py --procedure 'compare_power' --dim 25 --weight_pert 0.8 --std_pert 0.5 --model_instances 10
 --n_train 2000 --n_val 1000 --n_test 500 --n_runs 500 --n_boot 500 --stage_lambda`
 
 ## Comparison to KSD
 
 To compute KSD GoF hypothesis test power, run
 
`python driver.py --procedure 'ksd_power' --dim 25 --weight_pert 0.8 --std_pert 0.5 --n_runs 400 --n_boot 500 --n_train 0 --n_val 0 --n_test 500`

`python plotter.py --procedure 'ksd_power' --dim 25 --weight_pert 0.8 --std_pert 0.5 --n_runs 400 --n_boot 500 --n_train 0 --n_val 0 --n_test 500 --model_instances 1`

Then, KSD power and duration results can be compared to neural Stein results using

`python plotter.py --procedure 'compare_ksd' --dim 25 --weight_pert 0.8 --std_pert 0.5 --n_runs 400 --n_boot 500
--model_instances 5 --stage_lambda --lam_init 0.4 --beta 0.85 --lam_term 5e-4`
 
 ## MNIST Experiment
 
 The comparison of MSE in the MNIST experiments are conducted in a similar manner to the high-dimensional Gaussian mixture experiments, by including the `--mnist` flag and the `--mix_prop X` field.
 Note, `fit_rbm.py` must be run before the MNIST analysis can be conducted.
 Running `python driver.py --procedure "mnist_examine"` with appropriate flags identifying a trained network can be used to produce the embedding plots and localize the departures in distribution.
