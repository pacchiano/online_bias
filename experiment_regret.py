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
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
import numpy.random as npr
from scipy.stats import wasserstein_distance, ks_2samp
# from sklearn.linear_model import LogisticRegression

import torch

from datasets import *
from models import *

import IPython



def run_regret_experiment_pytorch( dataset, 
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
    random_init = True ,
    baseline_steps = 10000,
    baseline_batch_size =10,
    regret_wrt_baseline = True,
    MLP = True, 
    representation_layer_size = 10,
    mahalanobis_discount_factor = 1,
    verbose = True):


  protected_datasets_train, protected_datasets_test, train_dataset, test_dataset = get_dataset(dataset, batch_size, 1000)
  baseline_model = TorchBinaryLogisticRegression(random_init = random_init, fit_intercept=True, alpha = alpha, 
                MLP = MLP, representation_layer_size = representation_layer_size)
  
  if dataset == "MNIST":
    baseline_batch_size = batch_size


  for i in range(baseline_steps):
    print(i)
    global_batch, protected_batches = get_batches( protected_datasets_train, train_dataset, baseline_batch_size) 
    batch_X, batch_y = global_batch


    if i ==0:
      baseline_model.initialize_model(batch_X)           
      optimizer_baseline = torch.optim.SGD(baseline_model.network.parameters(), lr = 0.01)



    optimizer_baseline.zero_grad()
    loss_baseline = baseline_model.get_loss(batch_X, batch_y)
    

    loss_baseline.backward()
    optimizer_baseline.step()


  # IPython.embed()
  # raise ValueError("asldfkm")

  with torch.no_grad():
    baseline_batch_test, protected_batches_test = get_batches(protected_datasets_test, test_dataset, 1000)
    baseline_accuracy, protected_accuracies = get_accuracies(baseline_batch_test, protected_batches_test, baseline_model, threshold)
    loss_validation_baseline = baseline_model.get_loss(baseline_batch_test[0], baseline_batch_test[1])


  # import IPython
  # IPython.embed()
  # raise ValueError("alskdfm")



  num_protected_groups = len(protected_datasets_train)
  wass_distances = [[] for _ in range(num_protected_groups)]
  logging_counters = [[] for _ in range(num_protected_groups)]
  accuracies_list = []
  protected_accuracies_list = [[] for _ in range(num_protected_groups)]
  biased_accuracies_list = []
  biased_protected_accuracies_list = [[] for _ in range(num_protected_groups)]

  loss_validation = []
  loss_validation_biased = []

  train_regret = []

  losses = []
  accuracies = []

  counter = 0
  biased_data_totals = 0

  colors = ["red", "green", "violet", "orange"]

  model =  TorchBinaryLogisticRegression(random_init = random_init, fit_intercept=True, alpha = alpha, MLP = MLP, representation_layer_size = representation_layer_size)
  model_biased = TorchBinaryLogisticRegression(random_init = random_init, fit_intercept=True, alpha = alpha, MLP = MLP, representation_layer_size =representation_layer_size)

  cummulative_data_covariance = [] 
  inverse_cummulative_data_covariance = []

  #train_accuracies = []
  train_accuracies_biased = []
  timesteps = []

  #IPython.embed()

  while counter < max_num_steps:
    counter += 1 

    ### Start of the logistic steps
    global_batch, protected_batches = get_batches( protected_datasets_train, train_dataset, batch_size) 
    batch_X, batch_y = global_batch
    #global_prediction, protected_predictions = get_predictions(global_batch, protected_batches, model)
    if counter ==1:
      model.initialize_model(batch_X)
      model_biased.initialize_model(batch_X)


      optimizer_model = torch.optim.Adam(model.network.parameters(), lr = 0.01)
      optimizer_biased = torch.optim.Adam(model_biased.network.parameters(), lr = 0.01)


    # print("before optimization")
    # for parameter in model.network.parameters():
    #   print(torch.norm(parameter))


    optimizer_model.zero_grad()
    loss = model.get_loss(batch_X, batch_y)
    loss.backward()
    optimizer_model.step()


    # print("after optimization")
    # for parameter in model.network.parameters():
    #   print(torch.norm(parameter))

    
    #model.theta.detach()


    # IPython.embed()
    # raise ValueError("alskdfm")


    #logistic_gradient = model.get_gradient(batch_X, batch_y)
    #print("Logist gradient norm {}".format(np.linalg.norm(logistic_gradient)))
    #grad = logistic_learning_rate*logistic_gradient  
    #model.update(grad, 1.0)

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
      # print("############")
      # print("global biased prediction ", global_biased_prediction[i])
      # print("accept point ", accept_point)
      # print("batch y i ", batch_y[i])
      # print("outcome ", accept_point == batch_y[i])

      biased_train_accuracy += (accept_point == batch_y[i])*1.0 
      # import IPython
      # IPython.embed()
      # raise ValueError("asldkfm")
      if regret_wrt_baseline:      
        # import IPython
        # IPython.embed()
        # raise ValueError("laksdmf")
        batch_regret += baseline_accuracy - (accept_point == batch_y[i])*1.0

      else:
        if accept_point and batch_y[i] == 0:
          batch_regret += 1
        elif not accept_point and batch_y[i] == 1:
          batch_regret += 1.0

      if accept_point:
        #inverse_probabilities.append(1.0)#/global_biased_prediction[i])
        biased_batch_X.append(batch_X[i])
        biased_batch_y.append(batch_y[i])
        biased_batch_size += 1
    #print("biased batch size ")
    #print("biased batch size ", biased_batch_size)
    biased_batch_X = np.array(biased_batch_X)
    biased_batch_y = np.array(biased_batch_y)
    # print(biased_train_accuracy)
    # IPython.embed()
    # raise ValueError("asdlfkm")

    biased_train_accuracy = biased_train_accuracy/len(global_biased_prediction)
    batch_regret = batch_regret/len(global_biased_prediction)*1.0




    #inverse_probabilities = np.array(inverse_probabilities)

    biased_data_totals += biased_batch_size
    ### Train biased model on biased data




    if biased_batch_size > 0:

      # logistic_biased_gradient = model_biased.get_gradient(biased_batch_X, biased_batch_y,inverse_probabilities )
      # biased_grad = logistic_learning_rate*logistic_biased_gradient
      # model_biased.update(biased_grad, 1.0)

      optimizer_biased.zero_grad()
      IPython.embed()
      #raise ValueError("asdlfkm")
      biased_loss = model_biased.get_loss(biased_batch_X, biased_batch_y)
      biased_loss.backward()
      optimizer_biased.step()
      #model_biased.theta.detach()

      #updated_batch_X = model_biased.update_batch(biased_batch_X)
      representation_X =  model_biased.get_representation(biased_batch_X).detach()
      # IPython.embed()
      # raise ValueError("asdlkfm")
      representation_X = representation_X.numpy()
      # IPython.embed()
      # raise ValueError("asdflkm")
      if adjust_mahalanobis:
        if len(cummulative_data_covariance) == 0:
          cummulative_data_covariance = np.dot(np.transpose(representation_X), representation_X)
        else:
          cummulative_data_covariance = mahalanobis_discount_factor*cummulative_data_covariance +  np.dot(np.transpose(representation_X), representation_X)

      #### This can be done instead by using the Sherman-Morrison Formula.
        inverse_cummulative_data_covariance = torch.from_numpy(np.linalg.inv(mahalanobis_regularizer*np.eye(representation_X.shape[1])+ cummulative_data_covariance)).float()


    
    ## Compute accuracy diagnostics
    if counter % logging_frequency*1.0 == 0:
      train_regret.append(batch_regret)
      train_accuracies_biased.append(biased_train_accuracy)
      timesteps.append(counter)
      global_batch_test, protected_batches_test = get_batches(protected_datasets_test, test_dataset, 1000) 
      batch_X_test, batch_y_test = global_batch_test
      global_probabilities_list, protected_predictions = get_predictions(global_batch_test, protected_batches_test, model)
      total_accuracy, protected_accuracies = get_accuracies(global_batch_test, protected_batches_test, model, threshold)


      #### Compute loss diagnostics


      biased_loss = model_biased.get_loss(batch_X_test, batch_y_test)

      loss = model.get_loss(batch_X_test, batch_y_test)

      loss_validation.append(loss.detach())
      loss_validation_biased.append(biased_loss.detach())



      accuracies_list.append(total_accuracy)

      biased_global_probabilities_list, biased_protected_predictions = get_predictions(global_batch_test, protected_batches_test, model_biased)
      biased_total_accuracy, biased_protected_accuracies = get_accuracies(global_batch_test, protected_batches_test, model_biased, threshold)
      biased_accuracies_list.append(biased_total_accuracy)

      if verbose:
        print("Iteration {}".format(counter))
        print("Total proportion of biased data {}".format(1.0*biased_data_totals/(batch_size*counter)))

        ### Compute the global accuracy. 
        print("                                                               Accuracy ", total_accuracy)
                  
        ### Compute the global accuracy. 
        print("                                                               Biased Accuracy ", biased_total_accuracy)

      test_biased_accuracies_cum_averages = np.cumsum(biased_accuracies_list)
      test_biased_accuracies_cum_averages = test_biased_accuracies_cum_averages/(np.arange(len(timesteps))+1)
      accuracies_cum_averages = np.cumsum(accuracies_list)
      accuracies_cum_averages = accuracies_cum_averages/(np.arange(len(timesteps))+1)
      train_biased_accuracies_cum_averages = np.cumsum(train_accuracies_biased)
      train_biased_accuracies_cum_averages = train_biased_accuracies_cum_averages/(np.arange(len(timesteps))+1)
      train_cum_regret = np.cumsum(train_regret)
      #train_cum_regret = train_cum_regret/(np.arange(len(timesteps)) + 1)


  return timesteps, test_biased_accuracies_cum_averages, accuracies_cum_averages, train_biased_accuracies_cum_averages, train_cum_regret, loss_validation, loss_validation_biased, loss_validation_baseline, baseline_accuracy
















