from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib 
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats

import requests
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import ray
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy.random as npr
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.linear_model import LogisticRegression
import os
import pickle

from experiment_regret import *

ray.init()


@ray.remote
def run_experiment_parallel(dataset, logging_frequency, max_num_steps, logistic_learning_rate,threshold, biased_threshold, batch_size, 
	random_init, fit_intercept, mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha):

	timesteps, test_biased_accuracies_cum_averages, accuracies_cum_averages, train_biased_accuracies_cum_averages, train_cum_regret = run_regret_experiment( dataset, logging_frequency, max_num_steps, 
	              logistic_learning_rate, threshold, 
	              biased_threshold, batch_size, mahalanobis_regularizer, 
	              adjust_mahalanobis,epsilon_greedy, epsilon, alpha )
	return timesteps, test_biased_accuracies_cum_averages, accuracies_cum_averages, train_biased_accuracies_cum_averages, train_cum_regret

#@title Experiment parameters




def run_and_plot(dataset, logging_frequency, max_num_steps, logistic_learning_rate, threshold, 
	biased_threshold, batch_size, random_init, fit_intercept, num_experiments, mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha ):


	path = os.getcwd()
	if not os.path.isdir("{}/experiment_results/data/T{}".format(path, max_num_steps)):
		try:
			os.mkdir("{}/experiment_results/data/T{}".format(path, max_num_steps))
			os.mkdir("{}/experiment_results/figs/T{}".format(path, max_num_steps))
		except OSError:
			print("Creation of directories failed")
		else:
			print("Successfully created the directory")





	if epsilon_greedy and adjust_mahalanobis:
		raise ValueError("Both epsilon greedy and adjust mahalanobis are on!")

	experiment_summaries = [ run_experiment_parallel.remote( dataset, logging_frequency, max_num_steps, logistic_learning_rate,threshold, biased_threshold, batch_size, 
	random_init, fit_intercept, mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha ) for _ in range(num_experiments)]
	experiment_summaries = ray.get(experiment_summaries)


	timesteps = experiment_summaries[0][0]

	test_biased_accuracies_cum_averages_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))
	accuracies_cum_averages_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))
	train_biased_accuracies_cum_averages_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))
	train_cum_regret_summary = np.zeros((num_experiments, int(max_num_steps/logging_frequency)))

	for j in range(num_experiments):
		test_biased_accuracies_cum_averages_summary[j,:] = experiment_summaries[j][1]
		accuracies_cum_averages_summary[j,:] = experiment_summaries[j][2]
		train_biased_accuracies_cum_averages_summary[j,:] = experiment_summaries[j][3]
		train_cum_regret_summary[j, :] = experiment_summaries[j][4]

	mean_test_biased_accuracies_cum_averages = np.mean(test_biased_accuracies_cum_averages_summary, axis = 0)
	std_test_biased_accuracies_cum_averages = np.std(test_biased_accuracies_cum_averages_summary, axis = 0)

	mean_accuracies_cum_averages = np.mean(accuracies_cum_averages_summary, axis = 0)
	std_accuracies_cum_averages = np.std(accuracies_cum_averages_summary, axis =0 )

	mean_train_biased_accuracies_cum_averages = np.mean(train_biased_accuracies_cum_averages_summary, axis = 0)
	std_train_biased_accuracies_cum_averages = np.std(train_biased_accuracies_cum_averages_summary, axis = 0)


	mean_train_cum_regret_averages = np.mean(train_cum_regret_summary, axis = 0)
	std_train_cum_regret_averages = np.std(train_cum_regret_summary, axis = 0)




	plt.plot(timesteps,mean_test_biased_accuracies_cum_averages, label = "Online Decision Test - no optimism", linestyle = "dashed", linewidth = 3.5, color="blue")
	plt.fill_between(timesteps, mean_test_biased_accuracies_cum_averages - .5*std_test_biased_accuracies_cum_averages, 
		mean_test_biased_accuracies_cum_averages + .5*std_test_biased_accuracies_cum_averages, color = "blue", alpha = .1 )

	plt.plot(timesteps,mean_accuracies_cum_averages, label = "All data SGD Test", linestyle = "dashed", linewidth = 3.5, color = "red")
	plt.fill_between(timesteps, mean_accuracies_cum_averages - .5*std_accuracies_cum_averages, 
		mean_accuracies_cum_averages + .5*std_accuracies_cum_averages,color = "red", alpha = .1 )

	plt.plot(timesteps, mean_train_biased_accuracies_cum_averages, label = "Online Decision Train", linestyle = "dashed", linewidth = 3.5, color = "violet"  )
	plt.fill_between(timesteps, mean_train_biased_accuracies_cum_averages+ .5*std_train_biased_accuracies_cum_averages, 
		mean_train_biased_accuracies_cum_averages - .5*std_train_biased_accuracies_cum_averages, color = "violet", alpha = .1)


	if epsilon_greedy:
		plt.title("Test and Train Accuracies {} - Epsilon Greedy {}".format(dataset, epsilon))
		plot_name = "{}_test_train_accuracies_epsgreedy_{}".format(dataset, epsilon)
	if adjust_mahalanobis:
		plt.title("Test and Train Accuracies {} - Optimism alpha {} - Mreg {}".format(dataset, alpha, mahalanobis_regularizer))
		plot_name = "{}_test_train_accuracies_optimism_alpha_{}_mahreg_{}".format(dataset, alpha, mahalanobis_regularizer)
	if not epsilon_greedy and not adjust_mahalanobis:
		plt.title("Test and Train Accuracies {} ".format(dataset))
		plot_name  = "{}_test_train_accuracies_biased".format(dataset)

	plt.xlabel("Timesteps")
	plt.ylabel("Accuracy")
	plt.legend(loc = "lower right")
	plt.savefig("./experiment_results/figs/T{}/{}.png".format(max_num_steps, plot_name))
	plt.close('all')


	plt.plot(timesteps, mean_train_cum_regret_averages, label = "Regret", linestyle = "dashed", linewidth = 3.5, color = "blue")
	plt.fill_between(timesteps, mean_train_cum_regret_averages - .5*std_train_cum_regret_averages, 
		mean_train_cum_regret_averages + .5*std_train_cum_regret_averages, color = "blue", alpha = .1)
	plot_name = "{}_regret".format(plot_name)

	plt.xlabel("Timesteps")
	plt.ylabel("Regret")
	plt.legend(loc = "lower right")
	plt.savefig("./experiment_results/figs/T{}/{}.png".format(max_num_steps, plot_name))
	plt.close('all')


	pickle.dump((timesteps, mean_test_biased_accuracies_cum_averages, std_test_biased_accuracies_cum_averages, mean_accuracies_cum_averages, std_accuracies_cum_averages, 
		mean_train_biased_accuracies_cum_averages, std_train_biased_accuracies_cum_averages, 
		max_num_steps), open("./experiment_results/data/T{}/{}.p".format(max_num_steps, plot_name), "wb"))



def main():
	dataset = "Adult"
	logging_frequency = 10
	max_num_steps = 50000
	logistic_learning_rate = .01
	threshold = .5
	biased_threshold = .5
	batch_size = 10
	random_init = True
	fit_intercept = True

	num_experiments = 20

	adjust_mahalanobis = False
	mahalanobis_regularizer = .1
	alpha = 0

	epsilon_greedy = False
	epsilon = .1

	# ### run without any optimism or epsilon greedy
	# run_and_plot(dataset, logging_frequency, max_num_steps, logistic_learning_rate, threshold, 
	# 			biased_threshold, batch_size, random_init, fit_intercept, num_experiments, 
	# 			mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha)

	for alpha in [3,4]:
		adjust_mahalanobis = True
		epsilon_greedy = False
		for mahalanobis_regularizer in [1]:#[.1, 1]:
			run_and_plot(dataset, logging_frequency, max_num_steps, logistic_learning_rate, threshold, 
				biased_threshold, batch_size, random_init, fit_intercept, num_experiments, 
				mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha)

	# for epsilon in [.2]:#, .2, .5]:	
	# 	adjust_mahalanobis = False
	# 	epsilon_greedy = True
	# 	run_and_plot(dataset, logging_frequency, max_num_steps, logistic_learning_rate, threshold, 
	# 		biased_threshold, batch_size, random_init, fit_intercept, num_experiments, 
	# 		mahalanobis_regularizer, adjust_mahalanobis, epsilon_greedy, epsilon, alpha)



main()