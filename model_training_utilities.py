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







