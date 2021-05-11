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


def run_experiment(
    dataset,
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
):

    dataset = "MultiSVM"
    (
        protected_datasets_train,
        protected_datasets_test,
        train_dataset,
        test_dataset,
    ) = get_dataset(dataset)
    wass_distances = [[] for _ in range(len(PROTECTED_GROUPS))]
    logging_counters = [[] for _ in range(len(PROTECTED_GROUPS))]
    accuracies_list = []
    protected_accuracies_list = [[] for _ in range(len(PROTECTED_GROUPS))]
    biased_accuracies_list = []
    biased_protected_accuracies_list = [[] for _ in range(len(PROTECTED_GROUPS))]

    losses = []
    accuracies = []

    counter = 0
    biased_data_totals = 0

    colors = ["red", "green", "violet", "orange"]

    model = BinaryLogisticRegression(
        random_init=random_init, fit_intercept=True, alpha=alpha
    )
    model_biased = BinaryLogisticRegression(
        random_init=random_init, fit_intercept=True, alpha=alpha
    )

    cummulative_data_covariance = []
    inverse_cummulative_data_covariance = []

    while counter < max_num_steps:
        counter += 1

        ### Start of the logistic steps
        global_batch, protected_batches = get_batches(
            protected_datasets_train, train_dataset, batch_size
        )
        batch_X, batch_y = global_batch
        global_prediction, protected_predictions = get_predictions(
            global_batch, protected_batches, model
        )
        # print(batch_X, batch_y)
        logistic_gradient = model.get_gradient(batch_X, batch_y)
        # print("Logist gradient norm {}".format(np.linalg.norm(logistic_gradient)))
        grad = logistic_learning_rate * logistic_gradient
        model.update(grad, 1.0)

        ## Training biased model
        global_biased_prediction, protected_biased_predictions = get_predictions(
            global_batch,
            protected_batches,
            model_biased,
            inverse_cummulative_data_covariance,
        )
        biased_batch_X = []
        biased_batch_y = []
        inverse_probabilities = []
        biased_batch_size = 0
        # print("Global biased predictions ", global_biased_prediction)
        for i in range(len(global_biased_prediction)):
            # if np.random.random() <= global_biased_prediction[i]:
            if global_biased_prediction[i] > biased_threshold or (
                epsilon_greedy and np.random.random() < epsilon
            ):
                inverse_probabilities.append(1.0)  # /global_biased_prediction[i])
                biased_batch_X.append(batch_X[i])
                biased_batch_y.append(batch_y[i])
                biased_batch_size += 1
        biased_batch_X = np.array(biased_batch_X)
        biased_batch_y = np.array(biased_batch_y)
        inverse_probabilities = np.array(inverse_probabilities)

        biased_data_totals += biased_batch_size
        ### Train biased model on biased data
        if biased_batch_size > 0:

            logistic_biased_gradient = model_biased.get_gradient(
                biased_batch_X, biased_batch_y, inverse_probabilities
            )
            biased_grad = logistic_learning_rate * logistic_biased_gradient
            model_biased.update(biased_grad, 1.0)

            updated_batch_X = model.update_batch(biased_batch_X)

            if adjust_mahalanobis:
                if len(cummulative_data_covariance) == 0:
                    cummulative_data_covariance = mahalanobis_regularizer * np.eye(
                        updated_batch_X.shape[1]
                    ) + np.dot(np.transpose(updated_batch_X), updated_batch_X)
                else:
                    cummulative_data_covariance += np.dot(
                        np.transpose(updated_batch_X), updated_batch_X
                    )

                #### This can be done instead by using the Sherman-Morrison Formula.
                inverse_cummulative_data_covariance = np.linalg.inv(
                    cummulative_data_covariance
                )

        ## Compute accuracy diagnostics.
        if counter % logging_frequency == 0:
            print("Iteration {}".format(counter))
            print(
                "Total proportion of biased data {}".format(
                    1.0 * biased_data_totals / (batch_size * counter)
                )
            )

            global_batch_test, protected_batches_test = get_batches(
                protected_datasets_test, test_dataset, 100000000
            )
            batch_X_test, batch_y_test = global_batch_test
            global_probabilities_list, protected_predictions = get_predictions(
                global_batch_test, protected_batches_test, model
            )
            total_accuracy, protected_accuracies = get_accuracies(
                global_batch_test, protected_batches_test, model, threshold
            )
            ### Compute the global accuracy.
            print(
                "                                                               Accuracy ",
                total_accuracy,
            )

            accuracies_list.append(total_accuracy)
            protected_accuracies_list = [
                protected_accuracies_list[i] + [protected_accuracies[i]]
                for i in range(len(PROTECTED_GROUPS))
            ]
            plt.figure(figsize=(18, 16))
            plt.ylim(0.2, 1)
            plt.plot(accuracies_list, label="Global", linewidth=3.5, color="blue")
            [
                plt.plot(
                    protected_accuracies_list[i],
                    label=PROTECTED_GROUPS[i],
                    linewidth=3.5,
                    color=colors[i],
                )
                for i in range(len(PROTECTED_GROUPS))
            ]

            (
                biased_global_probabilities_list,
                biased_protected_predictions,
            ) = get_predictions(global_batch_test, protected_batches_test, model_biased)
            biased_total_accuracy, biased_protected_accuracies = get_accuracies(
                global_batch_test, protected_batches_test, model_biased, threshold
            )
            ### Compute the global accuracy.
            print(
                "                                                               Biased Accuracy ",
                biased_total_accuracy,
            )

            biased_accuracies_list.append(biased_total_accuracy)
            biased_protected_accuracies_list = [
                biased_protected_accuracies_list[i] + [biased_protected_accuracies[i]]
                for i in range(len(PROTECTED_GROUPS))
            ]

            plt.plot(
                biased_accuracies_list,
                label="Global Biased",
                linestyle="dashed",
                linewidth=3.5,
                color="blue",
            )
            [
                plt.plot(
                    biased_protected_accuracies_list[i],
                    label="Biased {}".format(PROTECTED_GROUPS[i]),
                    linestyle="dashed",
                    linewidth=3.5,
                    color=colors[i],
                )
                for i in range(len(PROTECTED_GROUPS))
            ]

            plt.title("Accuracies")
            plt.xlabel("Timesteps")
            plt.ylabel("Accuracy")
            plt.legend(loc="lower right")
            plt.show()

            plt.figure(figsize=(24, 5))

            for i in range(len(PROTECTED_GROUPS)):
                protected_group = PROTECTED_GROUPS[i]

                probabilities_list = protected_predictions[i]
                protected_accuracy = protected_accuracies[i]

                logging_counters[i].append(counter)

                print(
                    "                                                               Accuracy {} ".format(
                        PROTECTED_GROUPS[i]
                    ),
                    protected_accuracy,
                )
                plt.subplot(1, len(PROTECTED_GROUPS), i + 1)

                _, _, _ = plt.hist(
                    global_probabilities_list,
                    50,
                    density=True,
                    facecolor="g",
                    alpha=0.75,
                    label="global",
                )
                plt.xlabel("ys")
                plt.ylabel("Probability")
                plt.title("Histogram of responses {}.".format(protected_group))
                _, _, _ = plt.hist(
                    probabilities_list,
                    50,
                    density=True,
                    facecolor="r",
                    alpha=0.75,
                    label=protected_group,
                )

                plt.grid(True)
                plt.legend(loc="upper right")

            plt.show()

            plt.figure(figsize=(24, 5))

            for i in range(len(PROTECTED_GROUPS)):
                protected_group = PROTECTED_GROUPS[i]

                biased_probabilities_list = biased_protected_predictions[i]
                biased_protected_accuracy = biased_protected_accuracies[i]

                print(
                    "                                                              Biased Accuracy {} ".format(
                        PROTECTED_GROUPS[i]
                    ),
                    biased_protected_accuracy,
                )
                plt.subplot(1, len(PROTECTED_GROUPS), i + 1)

                _, _, _ = plt.hist(
                    biased_global_probabilities_list,
                    50,
                    density=True,
                    facecolor="g",
                    alpha=0.75,
                    label="global",
                    color="orange",
                )
                plt.xlabel("ys")
                plt.ylabel("Probability")
                plt.title("Biased Histogram of responses {}.".format(protected_group))
                _, _, _ = plt.hist(
                    biased_probabilities_list,
                    50,
                    density=True,
                    facecolor="r",
                    alpha=0.75,
                    label=protected_group,
                    color="blue",
                )

                plt.grid(True)
                plt.legend(loc="upper right")

            plt.show()

            if dataset == "SVM" or dataset == "MultiSVM":
                print(
                    "Groups ",
                    PROTECTED_GROUPS,
                    " Probabilities ",
                    probabilities,
                    " Variances ",
                    variances,
                    " Class list ",
                    class_list_per_center,
                )
                if d == 2:
                    plt.figure(figsize=(24, 5))
                    plt.subplot(1, 2, 1)
                    plt.title("Datapoints and classifier")
                    test_dataset.plot(
                        batch_size=1000, model=model, names=PROTECTED_GROUPS
                    )
                    plt.subplot(1, 2, 2)
                    plt.title("Biased datapoints and classifier")
                    test_dataset.plot(
                        batch_size=1000, model=model_biased, names=PROTECTED_GROUPS
                    )
                    plt.show()
