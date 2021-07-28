import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import random


import IPython

import sys

from dataclasses import dataclass
import torch
from torchvision import datasets, transforms
from typing import Any

import ray
from datasets import get_batches, get_dataset_simple, GrowingNumpyDataSet
from models import (
    TorchBinaryLogisticRegression,
    get_predictions,
    get_accuracies,
    get_accuracies_simple,
    get_breakdown_no_model,
    get_error_breakdown,
    get_special_breakdown
)

from pytorch_experiments import analyze_experiments, algo_to_params, NNParams, LinearModelHparams, ExplorationHparams, ExperimentResults



#@title Useful model training utilities
#Useful utilities. The following function allows to train a model 
### for a number of steps.
def gradient_step(model, optimizer, batch_X, batch_y):

    optimizer.zero_grad()
    loss = model.get_loss(batch_X, batch_y)
    loss.backward()
    optimizer.step()

    return model, optimizer


# def train_model(
#     model,
#     num_steps,
#     train_dataset,
#     batch_size,
#     verbose=False,
#     restart_model_full_minimization=True,
#     weight_decay=0.0
# ):
#     for i in range(num_steps):
#         if verbose:
#             print("train model iteration ", i)
#         batch_X, batch_y = train_dataset.get_batch(batch_size)
#         if i == 0:
#             restart_model_full_minimization = False
#             if restart_model_full_minimization:
#                 if len(batch_X.shape) == 1:
#                     batch_X = np.expand_dims(batch_X, axis=1)
#                 model.initialize_model(batch_X.shape[1])
#             optimizer = torch.optim.Adam(
#                 model.network.parameters(), lr=0.01, weight_decay=weight_decay
#             )

#         model, optimizer = gradient_step(model, optimizer, batch_X, batch_y)

#     return model


def train_model(
    model,
    num_steps,
    train_dataset,
    batch_size,
    verbose=False,
    restart_model_full_minimization=False,
    weight_decay=0.0
):
    if restart_model_full_minimization: 
        model.reinitialize_model()

    optimizer = torch.optim.Adam(model.network.parameters(), lr=0.01, weight_decay=weight_decay )

    for i in range(num_steps):
        if verbose:
            print("train model iteration ", i)
        batch_X, batch_y = train_dataset.get_batch(batch_size)

        model, optimizer = gradient_step(model, optimizer, batch_X, batch_y)

    return model








def train_baseline(dataset, num_timesteps, batch_size, MLP = True, 
    representation_layer_size = 10, threshold = .5, fit_intercept = True):
    

    (
        train_dataset,
        test_dataset,
    ) = get_dataset_simple(
        dataset=dataset,
        batch_size=batch_size,
        test_batch_size=10000000, 
        fit_intercept = fit_intercept)

    dataset_dimension = train_dataset.dimension

    
    baseline_model = TorchBinaryLogisticRegression(
        random_init=True,
        alpha=0,
        MLP=MLP,
        representation_layer_size=representation_layer_size,
        dim = train_dataset.dimension

    )


    baseline_model = train_model(
        baseline_model, num_timesteps, train_dataset, batch_size
    )



    print("Finished training baseline model")

    with torch.no_grad():
        baseline_batch_test = test_dataset.get_batch(10000000000) 
        
        baseline_test_accuracy = get_accuracies_simple(
            baseline_batch_test,
            baseline_model,
            threshold,
        )
        loss_validation_baseline = baseline_model.get_loss(
            baseline_batch_test[0], baseline_batch_test[1]
        )

    print("Baseline model accuracy {}".format(baseline_test_accuracy))

    return baseline_test_accuracy, baseline_model



def train_epsilon_greedy(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size, MLP = True, 
    representation_layer_size = 10, threshold = .5, epsilon = .1,
    verbose = False, fit_intercept = True, decaying_epsilon = False, 
    restart_model_full_minimization = False):
    
    (
        train_dataset,
        test_dataset,
    ) = get_dataset_simple(
        dataset=dataset,
        batch_size=batch_size,
        test_batch_size=10000000, 
        fit_intercept = True)

    dataset_dimension = train_dataset.dimension

    
    model = TorchBinaryLogisticRegression(
        random_init=True,
        alpha=0,
        MLP=MLP,
        representation_layer_size=representation_layer_size,
        dim = train_dataset.dimension
    )

    growing_training_dataset = GrowingNumpyDataSet()
    instantaneous_regrets = []
    instantaneous_accuracies = []
    eps_multiplier = 1.0

    for i in range(num_batches):
        if verbose:
            print("Processing epsilon greedy batch ", i)
        batch_X, batch_y = train_dataset.get_batch(batch_size)

        with torch.no_grad():
            
            if decaying_epsilon:
                eps_multiplier = 1.0/(np.sqrt(i+1))

            predictions = model.get_thresholded_predictions(batch_X, threshold)
            baseline_predictions = baseline_model.get_thresholded_predictions(batch_X, threshold)

            epsilon_greedy_mask = torch.bernoulli(torch.ones(predictions.shape)*epsilon*eps_multiplier).bool()
            mask = torch.max(epsilon_greedy_mask,predictions)

            boolean_labels_y = batch_y.bool()
            accuracy = (torch.sum(predictions*boolean_labels_y) +torch.sum( ~predictions*~boolean_labels_y))*1.0/batch_size
           
            accuracy_baseline = (torch.sum(baseline_predictions*boolean_labels_y) +torch.sum( ~baseline_predictions*~boolean_labels_y))*1.0/batch_size
            instantaneous_regret = accuracy_baseline - accuracy

            instantaneous_regrets.append(instantaneous_regret)
            instantaneous_accuracies.append(accuracy)


            filtered_batch_X = batch_X[mask, :]
            filtered_batch_y = batch_y[mask]


        growing_training_dataset.add_data(filtered_batch_X, filtered_batch_y)

        #### Filter the batch using the predictions
        #### Add the accepted points and their labels to the growing training dataset

        model = train_model( model, num_opt_steps, 
                growing_training_dataset, opt_batch_size, 
                restart_model_full_minimization = restart_model_full_minimization)

                
    print("Finished training epsilon-greedy model")

    with torch.no_grad():
        batch_test = test_dataset.get_batch(10000000000) 
        
        test_accuracy = get_accuracies_simple(
            batch_test,
            model,
            threshold,
        )

        loss_validation = model.get_loss(
            batch_test[0], batch_test[1]
        )

    print("Final test model accuracy {}".format(test_accuracy))

    return instantaneous_regrets, instantaneous_accuracies, test_accuracy








def train_PLOT(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size, MLP = True, 
    representation_layer_size = 10, threshold = .5, epsilon = .1,
    verbose = False, fit_intercept = True, decaying_epsilon = False, 
    restart_model_full_minimization = False):
    
    (
        train_dataset,
        test_dataset,
    ) = get_dataset_simple(
        dataset=dataset,
        batch_size=batch_size,
        test_batch_size=10000000, 
        fit_intercept = True)

    dataset_dimension = train_dataset.dimension

    
    model = TorchBinaryLogisticRegression(
        random_init=True,
        alpha=0,
        MLP=MLP,
        representation_layer_size=representation_layer_size,
        dim = train_dataset.dimension
    )

    growing_training_dataset = GrowingNumpyDataSet()
    instantaneous_regrets = []
    instantaneous_accuracies = []
    eps_multiplier = 1.0

    for i in range(num_batches):
        if verbose:
            print("Processing PLOT batch ", i)
        batch_X, batch_y = train_dataset.get_batch(batch_size)

        with torch.no_grad():
            
            if decaying_epsilon:
                eps_multiplier = 1.0/(np.sqrt(i+1))

            mle_predictions = model.get_thresholded_predictions(batch_X, threshold)
            epsilon_greedy_mask = torch.bernoulli(torch.ones(mle_predictions.shape)*epsilon*eps_multiplier).bool()
            
            pseudo_label_filtered_mask = epsilon_greedy_mask*~mle_predictions

            ### Compute the pseudo_label_filtered_batch            
            pseudo_label_filtered_batch_X = batch_X[pseudo_label_filtered_mask, :]
            pseudo_labels = torch.ones(pseudo_label_filtered_batch_X.shape[0]).type(batch_y.dtype)

        ### If the pseudo label filtered batch is nonempty train pseudo-label model

        if pseudo_label_filtered_batch_X.shape[0] != 0:
            growing_training_dataset.add_data(pseudo_label_filtered_batch_X, pseudo_labels)

            model = train_model( model, num_opt_steps, 
                growing_training_dataset, opt_batch_size, 
                restart_model_full_minimization  = restart_model_full_minimization)

            if verbose:
                print("Trained pseudo-label model ")


            ### Restore the data buffer to its last state
            growing_training_dataset.pop_last_data()

        ### Figure the optimistic predictions 
        with torch.no_grad():            
            optimistic_predictions = model.get_thresholded_predictions(batch_X, threshold)
            baseline_predictions = baseline_model.get_thresholded_predictions(batch_X, threshold)

            boolean_labels_y = batch_y.bool()
            accuracy = (torch.sum(optimistic_predictions*boolean_labels_y) +torch.sum( ~optimistic_predictions*~boolean_labels_y))*1.0/batch_size
           
            accuracy_baseline = (torch.sum(baseline_predictions*boolean_labels_y) +torch.sum( ~baseline_predictions*~boolean_labels_y))*1.0/batch_size
            instantaneous_regret = accuracy_baseline - accuracy


            filtered_batch_X  = batch_X[optimistic_predictions, :]
            filtered_batch_y = batch_y[optimistic_predictions]


        instantaneous_regrets.append(instantaneous_regret)
        instantaneous_accuracies.append(accuracy)


        #### Filter the batch using the predictions
        #### Add the accepted points and their labels to the growing training dataset
        growing_training_dataset.add_data(filtered_batch_X, filtered_batch_y)

        ### Train MLE
        model = train_model( model, num_opt_steps, 
                growing_training_dataset, opt_batch_size, 
                restart_model_full_minimization  = restart_model_full_minimization)

                
    print("Finished training epsilon-greedy model")

    with torch.no_grad():
        batch_test = test_dataset.get_batch(10000000000) 
        
        test_accuracy = get_accuracies_simple(
            batch_test,
            model,
            threshold,
        )

        loss_validation = model.get_loss(
            batch_test[0], batch_test[1]
        )

    print("Final test model accuracy {}".format(test_accuracy))

    return instantaneous_regrets, instantaneous_accuracies, test_accuracy












dataset = "Bank"



baseline_test_accuracy, baseline_model = train_baseline(dataset, num_timesteps = 1000, 
    batch_size = 32, 
    MLP = True, representation_layer_size = 10)


instantaneous_epsilon_regrets, instantaneous_epsilon_accuracies, test_epsilon_accuracy = train_epsilon_greedy(dataset, baseline_model, 
    num_batches = 100, batch_size = 32, 
    num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
    representation_layer_size = 10, threshold = .5, verbose = True, decaying_epsilon = True)



instantaneous_PLOT_regrets, instantaneous_PLOT_accuracies, test_PLOT_accuracy = train_PLOT(dataset, baseline_model, 
    num_batches = 100, batch_size = 32, 
    num_opt_steps = 1000, opt_batch_size = 20, MLP = True, 
    representation_layer_size = 10, threshold = .5, verbose = True, decaying_epsilon = True)



IPython.embed()

