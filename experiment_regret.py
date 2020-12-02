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
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy.random as npr
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.linear_model import LogisticRegression

from datasets import *
from models import *

def run_regret_experiment( dataset, logging_frequency, max_num_steps, 
  logistic_learning_rate, threshold, 
  biased_threshold, batch_size, mahalanobis_regularizer, 
  adjust_mahalanobis,epsilon_greedy, epsilon, alpha, random_init = True ):


  protected_datasets_train, protected_datasets_test, train_dataset, test_dataset = get_dataset(dataset)
  wass_distances = [[] for _ in range(len(PROTECTED_GROUPS))]
  logging_counters = [[] for _ in range(len(PROTECTED_GROUPS))]
  accuracies_list = []
  protected_accuracies_list = [[] for _ in range(len(PROTECTED_GROUPS))]
  biased_accuracies_list = []
  biased_protected_accuracies_list = [[] for _ in range(len(PROTECTED_GROUPS))]

  train_regret = []

  losses = []
  accuracies = []

  counter = 0
  biased_data_totals = 0

  colors = ["red", "green", "violet", "orange"]

  model = BinaryLogisticRegression(random_init = random_init, fit_intercept=True, alpha = alpha)
  model_biased = BinaryLogisticRegression(random_init = random_init, fit_intercept=True, alpha = alpha)

  cummulative_data_covariance = [] 
  inverse_cummulative_data_covariance = []

  #train_accuracies = []
  train_accuracies_biased = []
  timesteps = []

  while counter < max_num_steps:
    counter += 1 

    ### Start of the logistic steps
    global_batch, protected_batches = get_batches( protected_datasets_train, train_dataset, batch_size) 
    batch_X, batch_y = global_batch
    global_prediction, protected_predictions = get_predictions(global_batch, protected_batches, model)
    #print(batch_X, batch_y)
    logistic_gradient = model.get_gradient(batch_X, batch_y)
    #print("Logist gradient norm {}".format(np.linalg.norm(logistic_gradient)))
    grad = logistic_learning_rate*logistic_gradient  
    model.update(grad, 1.0)

    ## Training biased model
    global_biased_prediction, protected_biased_predictions = get_predictions(global_batch, protected_batches, model_biased, inverse_cummulative_data_covariance)
    biased_batch_X = []
    biased_batch_y = []
    inverse_probabilities = []
    biased_batch_size = 0
    #print("Global biased predictions ", global_biased_prediction)
    biased_train_accuracy = 0
    batch_regret = 0
    for i in range(len(global_biased_prediction)):
      #if np.random.random() <= global_biased_prediction[i]:
      accept_point = global_biased_prediction[i] > biased_threshold or (epsilon_greedy and np.random.random() < epsilon)
      #print(accept_point, " ", batch_y[i], accept_point == batch_y[i])
      #if accept_point == 
      biased_train_accuracy += (accept_point == batch_y[i]) 
      if accept_point and batch_y[i] == 0:
        batch_regret += 1
      elif not accept_point and batch_y[i] == 1:
        batch_regret += 1.0

      if accept_point:
        inverse_probabilities.append(1.0)#/global_biased_prediction[i])
        biased_batch_X.append(batch_X[i])
        biased_batch_y.append(batch_y[i])
        biased_batch_size += 1
    #print("biased batch size ")
    #print("biased batch size ", biased_batch_size)
    biased_batch_X = np.array(biased_batch_X)
    biased_batch_y = np.array(biased_batch_y)
    biased_train_accuracy = biased_train_accuracy/len(global_biased_prediction)
    batch_regret = batch_regret/len(global_biased_prediction)*1.0




    inverse_probabilities = np.array(inverse_probabilities)

    biased_data_totals += biased_batch_size
    ### Train biased model on biased data
    if biased_batch_size > 0:

      logistic_biased_gradient = model_biased.get_gradient(biased_batch_X, biased_batch_y,inverse_probabilities )
      biased_grad = logistic_learning_rate*logistic_biased_gradient
      model_biased.update(biased_grad, 1.0)

      updated_batch_X = model.update_batch(biased_batch_X)

      if adjust_mahalanobis:
        if len(cummulative_data_covariance) == 0:
          cummulative_data_covariance = mahalanobis_regularizer*np.eye(updated_batch_X.shape[1])+np.dot(np.transpose(updated_batch_X), updated_batch_X)
        else:
          cummulative_data_covariance += np.dot(np.transpose(updated_batch_X), updated_batch_X)

      #### This can be done instead by using the Sherman-Morrison Formula.
        inverse_cummulative_data_covariance = np.linalg.inv(cummulative_data_covariance)

    
    ## Compute accuracy diagnostics
    if counter % logging_frequency*1.0 == 0:
      train_regret.append(batch_regret)
      train_accuracies_biased.append(biased_train_accuracy)
      timesteps.append(counter)
      global_batch_test, protected_batches_test = get_batches(protected_datasets_test, test_dataset, 100000000) 
      batch_X_test, batch_y_test = global_batch_test
      global_probabilities_list, protected_predictions = get_predictions(global_batch_test, protected_batches_test, model)
      total_accuracy, protected_accuracies = get_accuracies(global_batch_test, protected_batches_test, model, threshold)

      accuracies_list.append(total_accuracy)

      biased_global_probabilities_list, biased_protected_predictions = get_predictions(global_batch_test, protected_batches_test, model_biased)
      biased_total_accuracy, biased_protected_accuracies = get_accuracies(global_batch_test, protected_batches_test, model_biased, threshold)
      biased_accuracies_list.append(biased_total_accuracy)


    ## Compute accuracy diagnostics.
    #if counter % logging_frequency == 0:
      print("Iteration {}".format(counter))
      print("Total proportion of biased data {}".format(1.0*biased_data_totals/(batch_size*counter)))

      ### Compute the global accuracy. 
      print("                                                               Accuracy ", total_accuracy)

      # plt.figure(figsize=(10,11))
      # plt.ylim(.2, 1)
                
      ### Compute the global accuracy. 
      print("                                                               Biased Accuracy ", biased_total_accuracy)

      
      # plt.plot(timesteps,biased_accuracies_list, label = "Biased Test", linestyle = "dashed", linewidth = 3.5, color="blue")
      # plt.plot(timesteps,accuracies_list, label = "Unbiased", linestyle = "dashed", linewidth = 3.5, color = "red")
      # plt.plot(timesteps, train_accuracies_biased, label = "Biased Train", linestyle = "dashed", linewidth = 3.5, color = "violet"  )
      test_biased_accuracies_cum_averages = np.cumsum(biased_accuracies_list)
      test_biased_accuracies_cum_averages = test_biased_accuracies_cum_averages/(np.arange(len(timesteps))+1)
      accuracies_cum_averages = np.cumsum(accuracies_list)
      accuracies_cum_averages = accuracies_cum_averages/(np.arange(len(timesteps))+1)
      train_biased_accuracies_cum_averages = np.cumsum(train_accuracies_biased)
      train_biased_accuracies_cum_averages = train_biased_accuracies_cum_averages/(np.arange(len(timesteps))+1)
      train_cum_regret = np.cumsum(train_regret)
      #train_cum_regret = train_cum_regret/(np.arange(len(timesteps)) + 1)


  return timesteps, test_biased_accuracies_cum_averages, accuracies_cum_averages, train_biased_accuracies_cum_averages, train_cum_regret






# def plot_data():


#       plt.plot(timesteps,test_biased_accuracies_cum_averages, label = "Biased Test", linestyle = "dashed", linewidth = 3.5, color="blue")
#       plt.plot(timesteps,accuracies_cum_averages, label = "Unbiased Test", linestyle = "dashed", linewidth = 3.5, color = "red")
#       plt.plot(timesteps, train_biased_accuracies_cum_averages, label = "Biased Train", linestyle = "dashed", linewidth = 3.5, color = "violet"  )



#       plt.title("Test and Train Accuracies")
#       plt.xlabel("Timesteps")
#       plt.ylabel("Accuracy")
#       plt.legend(loc = "lower right")
#       plt.savefig("./figs/test_train_accuracies")
