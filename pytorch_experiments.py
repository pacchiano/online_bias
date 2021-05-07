from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib 
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats

import requests
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import ray
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
import numpy.random as npr
from scipy.stats import wasserstein_distance, ks_2samp
#from sklearn.linear_model import LogisticRegression
import os
import pickle
import itertools

from experiment_regret import *



USE_RAY = True#False
ray.init()
@ray.remote
def run_experiment_parallel(dataset, logging_frequency, max_num_steps, logistic_learning_rate,threshold, biased_threshold, batch_size, 
	random_init, fit_intercept, mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha, MLP, representation_layer_size, baseline_steps, mahalanobis_discount_factor, training_mode, decision_type):

	timesteps, test_biased_accuracies_cum_averages, accuracies_cum_averages, train_biased_accuracies_cum_averages, train_cum_regret, loss_validation, loss_validation_biased, loss_baseline, baseline_accuracy = run_regret_experiment_pytorch( dataset, 
																					logging_frequency, 
																					max_num_steps, 
																				    logistic_learning_rate, 
																				    threshold, 
																				    biased_threshold, 
																				    batch_size, 
																				    mahalanobis_regularizer,           
																				    adjust_mahalanobis,
																				    epsilon_greedy, 
																				    epsilon, 
																				    alpha, 
																				    MLP = MLP,
																				    representation_layer_size = representation_layer_size,
																				    baseline_steps = baseline_steps ,
																				    mahalanobis_discount_factor = mahalanobis_discount_factor,
																				    training_mode = training_mode,
																				    decision_type = decision_type)
	return timesteps, test_biased_accuracies_cum_averages, accuracies_cum_averages, train_biased_accuracies_cum_averages, train_cum_regret,loss_validation, loss_validation_biased, loss_baseline, baseline_accuracy





def run_and_plot(dataset, logging_frequency, max_num_steps, logistic_learning_rate, threshold, 
	biased_threshold, batch_size, random_init, fit_intercept, num_experiments, mahalanobis_regularizer, 
	adjust_mahalanobis, epsilon_greedy, epsilon, alpha, MLP, representation_layer_size, baseline_steps , mahalanobis_discount_factor, training_mode, decision_type):
	path = os.getcwd()

	network_type = "MLP{}".format(representation_layer_size) if MLP else "Linear"
	base_data_directory = "{}/experiment_results/T{}/{}/{}/data".format(path, max_num_steps, network_type, dataset)
	base_figs_directory = "{}/experiment_results/T{}/{}/{}/figs".format(path, max_num_steps, network_type, dataset)


	if not os.path.isdir(base_data_directory):
		try:

			#os.makedirs(base_figs_directory)
			os.makedirs(base_data_directory)
		except OSError:
			print("Creation of data directories failed")
		else:
			print("Successfully created the data directory")

	if not os.path.isdir(base_figs_directory):
		try:

			os.makedirs(base_figs_directory)
			#os.makedirs(base_data_directory)
		except OSError:
			print("Creation of figs directories failed")
		else:
			print("Successfully created the figs directory")



	if epsilon_greedy and adjust_mahalanobis:
		raise ValueError("Both epsilon greedy and adjust mahalanobis are on!")

	print("Starting Experiment {} T{} {} {} {} epsilongreedy {} epsilon {} adjmahalanobis {} mahreg {} alpha {}".format(dataset, max_num_steps,  training_mode, decision_type, network_type, epsilon_greedy, epsilon , adjust_mahalanobis, mahalanobis_regularizer, alpha ))
	# IPython.embed()
	# raise ValueError("asdlfkm")
	if USE_RAY:
		experiment_summaries = [ run_experiment_parallel.remote( dataset, logging_frequency, max_num_steps, logistic_learning_rate,threshold, biased_threshold, batch_size, 
		random_init, fit_intercept, mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha, MLP, representation_layer_size, baseline_steps, 
		mahalanobis_discount_factor, training_mode, decision_type ) for _ in range(num_experiments)]
		experiment_summaries = ray.get(experiment_summaries)

	else:
		experiment_summaries = [ run_experiment_parallel( dataset, logging_frequency, max_num_steps, logistic_learning_rate,threshold, biased_threshold, batch_size, 
		random_init, fit_intercept, mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha, MLP, representation_layer_size, baseline_steps, 
		mahalanobis_discount_factor, training_mode, decision_type ) for _ in range(num_experiments)]


	

	timesteps = experiment_summaries[0][0]

	test_biased_accuracies_cum_averages_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))
	accuracies_cum_averages_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))
	train_biased_accuracies_cum_averages_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))
	train_cum_regret_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))
	loss_validation_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))
	loss_validation_biased_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))

	loss_validation_baseline_summary = np.zeros(num_experiments)
	accuracy_validation_baseline_summary = np.zeros(num_experiments)

	for j in range(num_experiments):
		test_biased_accuracies_cum_averages_summary[j,:] = experiment_summaries[j][1]
		accuracies_cum_averages_summary[j,:] = experiment_summaries[j][2]
		train_biased_accuracies_cum_averages_summary[j,:] = experiment_summaries[j][3]
		train_cum_regret_summary[j, :] = experiment_summaries[j][4]
		loss_validation_summary[j,:] = experiment_summaries[j][5]
		loss_validation_biased_summary[j, :] = experiment_summaries[j][6]
		loss_validation_baseline_summary[j] = experiment_summaries[j][7]
		accuracy_validation_baseline_summary[j] = experiment_summaries[j][8]


	mean_test_biased_accuracies_cum_averages = np.mean(test_biased_accuracies_cum_averages_summary, axis = 0)
	std_test_biased_accuracies_cum_averages = np.std(test_biased_accuracies_cum_averages_summary, axis = 0)

	mean_accuracies_cum_averages = np.mean(accuracies_cum_averages_summary, axis = 0)
	std_accuracies_cum_averages = np.std(accuracies_cum_averages_summary, axis =0 )

	mean_train_biased_accuracies_cum_averages = np.mean(train_biased_accuracies_cum_averages_summary, axis = 0)
	std_train_biased_accuracies_cum_averages = np.std(train_biased_accuracies_cum_averages_summary, axis = 0)

	mean_train_cum_regret_averages = np.mean(train_cum_regret_summary, axis = 0)
	std_train_cum_regret_averages = np.std(train_cum_regret_summary, axis = 0)

	mean_loss_validation_averages = np.mean(loss_validation_summary, axis = 0)
	std_loss_validation_averages = np.std(loss_validation_summary, axis = 0)

	mean_loss_validation_biased_averages = np.mean(loss_validation_biased_summary, axis = 0)
	std_loss_validation_biased_averages = np.std(loss_validation_biased_summary, axis = 0)

	mean_loss_validation_baseline_summary = np.mean(loss_validation_baseline_summary)
	std_loss_validation_baseline_summary = np.std(loss_validation_baseline_summary)

	mean_accuracy_validation_baseline_summary = np.mean(accuracy_validation_baseline_summary)
	std_accuracy_validation_baseline_summary = np.std(accuracy_validation_baseline_summary)



	plt.plot(timesteps,mean_test_biased_accuracies_cum_averages, label = "Biased Model Test - no decision adjustment", linestyle = "dashed", linewidth = 3.5, color="blue")
	plt.fill_between(timesteps, mean_test_biased_accuracies_cum_averages - .5*std_test_biased_accuracies_cum_averages, 
		mean_test_biased_accuracies_cum_averages + .5*std_test_biased_accuracies_cum_averages, color = "blue", alpha = .1 )

	plt.plot(timesteps,mean_accuracies_cum_averages, label = "Unbiased Model Test - all data train", linestyle = "dashed", linewidth = 3.5, color = "red")
	plt.fill_between(timesteps, mean_accuracies_cum_averages - .5*std_accuracies_cum_averages, 
		mean_accuracies_cum_averages + .5*std_accuracies_cum_averages,color = "red", alpha = .1 )

	plt.plot(timesteps, mean_train_biased_accuracies_cum_averages, label = "Online Biased Model - filtered data train", linestyle = "dashed", linewidth = 3.5, color = "violet"  )
	plt.fill_between(timesteps, mean_train_biased_accuracies_cum_averages+ .5*std_train_biased_accuracies_cum_averages, 
		mean_train_biased_accuracies_cum_averages - .5*std_train_biased_accuracies_cum_averages, color = "violet", alpha = .1)


	plt.plot(timesteps, [mean_accuracy_validation_baseline_summary]*len(timesteps), label = "Baseline Accuracy", linestyle = "dashed", linewidth = 3.5, color = "black")
	plt.fill_between(timesteps, [mean_accuracy_validation_baseline_summary - .5*std_accuracy_validation_baseline_summary]*len(timesteps), 
		[mean_accuracy_validation_baseline_summary + .5*std_accuracy_validation_baseline_summary]*len(timesteps), color = "black", alpha = .1)


	if decision_type == "simple":
		if epsilon_greedy:
			plt.title("Test and Train Accuracies {} - Epsilon Greedy {} - {} - {}".format(dataset, epsilon, network_type, training_mode), fontsize = 8)
			plot_name = "{}_test_train_accuracies_epsgreedy_{}_{}_{}".format(dataset, epsilon, network_type, training_mode)
		if adjust_mahalanobis:
			plt.title("Test and Train Accuracies {} - Optimism alpha {} - Mreg {} - Mdisc {} - {} - {}".format(dataset, alpha, mahalanobis_regularizer, mahalanobis_discount_factor, network_type, training_mode), fontsize = 8)
			plot_name = "{}_test_train_accuracies_optimism_alpha_{}_mahreg_{}_mdisc_{}_{}_{}".format(dataset, alpha, mahalanobis_regularizer, mahalanobis_discount_factor, network_type, training_mode)
		if not epsilon_greedy and not adjust_mahalanobis:
			plt.title("Test and Train Accuracies {} - {} - {} ".format(dataset, network_type, training_mode), fontsize = 8)
			plot_name  = "{}_test_train_accuracies_biased_{}_{}".format(dataset, network_type, training_mode)
	elif decision_type == "counterfactual":
		plt.title("Test and Train Accuracies {} - {} - {} - {}".format(dataset, network_type, training_mode, decision_type), fontsize = 8)
		plot_name  = "{}_test_train_accuracies_biased_{}_{}_{}".format(dataset, network_type, training_mode, decision_type)

	else:
		raise ValueError("Decision type not recognized {}".format(decision_type))

	plt.xlabel("Timesteps")
	plt.ylabel("Accuracy")
	#plt.legend(loc = "lower right")

	lg = plt.legend(bbox_to_anchor=(1.05, 1), fontsize = 8,loc = "upper left")
	#plt.tight_layout()


	plt.savefig("{}/{}.png".format(base_figs_directory, plot_name), bbox_extra_artists=(lg,), bbox_inches='tight')
	plt.close('all')


	plt.plot(timesteps, mean_train_cum_regret_averages, label = "Regret", linestyle = "dashed", linewidth = 3.5, color = "blue")
	plt.fill_between(timesteps, mean_train_cum_regret_averages - .5*std_train_cum_regret_averages, 
		mean_train_cum_regret_averages + .5*std_train_cum_regret_averages, color = "blue", alpha = .1)
	
	if decision_type == "simple":
		if epsilon_greedy:
			plt.title("Regret {} - Epsilon Greedy {} - {} - {}".format(dataset, epsilon, network_type, training_mode), fontsize = 8)
			plot_name = "{}_regret_epsgreedy_{}_{}_{}".format(dataset, epsilon, network_type, training_mode)
		if adjust_mahalanobis:
			plt.title("Regret {} - Optimism alpha {} - Mreg {} - Mdisc {} - {} - {}".format(dataset, alpha, mahalanobis_regularizer, mahalanobis_discount_factor, network_type, training_mode), fontsize = 8)
			plot_name = "{}_regret_optimism_alpha_{}_mahreg_{}_mdisc_{}_{}_{}".format(dataset, alpha, mahalanobis_regularizer, mahalanobis_discount_factor, network_type, training_mode)
		if not epsilon_greedy and not adjust_mahalanobis:
			plt.title("Regret {} - {} - {}".format(dataset, network_type, training_mode), fontsize = 8)
			plot_name  = "{}_regret_biased_{}_{}".format(dataset, network_type, training_mode)
	elif decision_type == "counterfactual":
		plt.title("Regret {} - {} - {} - {}".format(dataset, network_type, training_mode, decision_type), fontsize = 8)
		plot_name  = "{}_regret_biased_{}_{}_{}".format(dataset, network_type, training_mode, decision_type)
	else:
		raise ValueError("Decision type not recognized {}".format(decision_type))


	plt.xlabel("Timesteps")
	plt.ylabel("Regret")
	#plt.legend(loc = "lower right")
	lg = plt.legend(bbox_to_anchor=(1.05, 1), fontsize = 8, loc = "upper left")
	#plt.tight_layout()

	plt.savefig("{}/{}.png".format(base_figs_directory, plot_name),bbox_extra_artists=(lg,), bbox_inches='tight')
	
	plt.close('all')
	### LOSS PLOTS
	plt.plot(timesteps, mean_loss_validation_averages, label = "Unbiased model loss", linestyle = "dashed", linewidth = 3.5, color = "blue")
	plt.fill_between(timesteps, mean_loss_validation_averages - .5*std_loss_validation_averages, 
		mean_loss_validation_averages + .5*std_loss_validation_averages, color = "blue", alpha = .1)

	plt.plot(timesteps, mean_loss_validation_biased_averages, label = "Biased model loss", linestyle = "dashed", linewidth = 3.5, color = "red")
	plt.fill_between(timesteps, mean_loss_validation_biased_averages - .5*std_loss_validation_biased_averages, 
		mean_loss_validation_biased_averages + .5*std_loss_validation_biased_averages, color = "red", alpha = .1)

	plt.plot(timesteps, [mean_loss_validation_baseline_summary]*len(timesteps), label = "Baseline Loss", linestyle = "dashed", linewidth = 3.5, color = "black")
	plt.fill_between(timesteps, [mean_loss_validation_baseline_summary - .5*std_loss_validation_baseline_summary]*len(timesteps), 
		[mean_loss_validation_baseline_summary + .5*std_loss_validation_baseline_summary]*len(timesteps), color = "black", alpha = .1)

	if decision_type == "simple":
		if epsilon_greedy:
			plt.title("Loss {} - Epsilon Greedy {} - {} - {}".format(dataset, epsilon, network_type, training_mode), fontsize = 8)
			plot_name = "{}_loss_epsgreedy_{}_{}_{}".format(dataset, epsilon, network_type, training_mode)
		if adjust_mahalanobis:
			plt.title("Loss {} - Optimism alpha {} - Mreg {} - Mdisc {} - {} - {}".format(dataset, alpha, mahalanobis_regularizer, mahalanobis_discount_factor, network_type, training_mode), fontsize = 8)
			plot_name = "{}_loss_optimism_alpha_{}_mahreg_{}_mdisc_{}_{}_{}".format(dataset, alpha, mahalanobis_regularizer, mahalanobis_discount_factor, network_type, training_mode)
		if not epsilon_greedy and not adjust_mahalanobis:
			plt.title("Loss {} - {} - {}".format(dataset, network_type, training_mode), fontsize = 8)
			plot_name  = "{}_loss_biased_{}_{}".format(dataset, network_type, training_mode)
	elif decision_type == "counterfactual":
		plt.title("Loss {} - {} - {} - {}".format(dataset, network_type, training_mode, decision_type), fontsize = 8)
		plot_name  = "{}_loss_biased_{}_{}_{}".format(dataset, network_type, training_mode, decision_type)


	plt.xlabel("Timesteps")
	plt.ylabel("Loss")
	#plt.legend(loc = "lower right")good

	lg = plt.legend(bbox_to_anchor=(1.05, 1), fontsize = 8, loc = "upper left")
	#plt.tight_layout()

	plt.savefig("{}/{}.png".format(base_figs_directory, plot_name), bbox_extra_artists=(lg,), bbox_inches='tight')
	plt.close('all')



	pickle.dump((timesteps, mean_test_biased_accuracies_cum_averages, std_test_biased_accuracies_cum_averages, mean_accuracies_cum_averages, std_accuracies_cum_averages, 
		mean_train_biased_accuracies_cum_averages, std_train_biased_accuracies_cum_averages, 
		max_num_steps, mean_loss_validation_averages, std_loss_validation_averages,mean_loss_validation_biased_averages, 
		std_loss_validation_biased_averages ), open("{}/{}.p".format(base_data_directory, plot_name), "wb"))



def main():

	for dataset in [  "MultiSVM", "Adult", "MNIST",]:
		training_modes = [ "full_minimization", "gradient_step"]#, "full_minimization"]

		logging_frequency = 10
		max_num_steps = 30
		baseline_steps = 10000
		logistic_learning_rate = .01
		threshold = .5
		biased_threshold = .5
		batch_size = 10
		random_init = True
		fit_intercept = True

		num_experiments = 5

		representation_layer_sizes = [10, 40, 100]
		mahalanobis_discount_factors = [1, .9, .8]
		mahalanobis_reguarizers = [.1]
		epsilons = [.1, .2, .5]
		alphas = [1, 4]

		MLP_and_layer_sizes = [(False, None)] + list(itertools.product([True], representation_layer_sizes))
		trainmodes_decision_type_adjmahalanobis_epsgreedy_epsilon_mahdisc_mahreg_alpha = list(itertools.product(training_modes, ["simple"], [False], [True], epsilons, [0], [0], [0] )) + list(itertools.product(training_modes, ["simple"], [True], [False], [0], mahalanobis_discount_factors, mahalanobis_reguarizers, alphas ))
		trainmodes_decision_type_adjmahalanobis_epsgreedy_epsilon_mahdisc_mahreg_alpha += [("full_minimization", "counterfactual", False, False, 0, 0, 0, 0) ]

		all_params = list(itertools.product(MLP_and_layer_sizes, trainmodes_decision_type_adjmahalanobis_epsgreedy_epsilon_mahdisc_mahreg_alpha))
		all_params = [tuple(list(x[0]) + list(x[1])) for x in all_params]


		for MLP, representation_layer_size, training_mode, decision_type, adjust_mahalanobis, epsilon_greedy, epsilon, mahalanobis_discount_factor, mahalanobis_regularizer, alpha in all_params:
			training_mode = "full_minimization"
			decision_type = "counterfactual"
			adjust_mahalanobis = False
			epsilon_greedy = False
			epsilon = 0
			MLP = False
			representation_layer_size = None


			run_and_plot(dataset, logging_frequency, max_num_steps, logistic_learning_rate, threshold, 
						biased_threshold, batch_size, random_init, fit_intercept, num_experiments, 
						mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha, MLP, representation_layer_size, baseline_steps, mahalanobis_discount_factor, training_mode, decision_type)

			raise ValueError("Asdlfkm")


main()
