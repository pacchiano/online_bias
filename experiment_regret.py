from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gc
import torch

from datasets import get_batches, get_dataset, GrowingNumpyDataSet
from models import (
    TorchBinaryLogisticRegression,
    get_predictions,
    get_accuracies,
    get_accuracies_simple,
)


def train_model(
    model,
    num_steps,
    train_dataset,
    batch_size,
    verbose=False,
    restart_model_full_minimization=True,
):

    for i in range(num_steps):
        if verbose:
            print("train model iteration ", i)
        batch_X, batch_y = train_dataset.get_batch(batch_size)

        if i == 0:
            if restart_model_full_minimization:
                model.initialize_model(batch_X.shape[1])
            # model.initialize_model(batch_X)
            optimizer = torch.optim.Adam(model.network.parameters(), lr=0.01)

        optimizer.zero_grad()
        loss = model.get_loss(batch_X, batch_y)

        loss.backward()
        optimizer.step()
    return model


def train_model_with_stopping(
    model,
    min_epoch_size,
    train_dataset,
    batch_size,
    verbose=False,
    restart_model_full_minimization=True,
    eps=0.0001,
    max_epochs=7,
    eps_epoch_cycle=30,
):
    curr_epoch_size = min_epoch_size
    prev_loss_value = float("inf")

    _, min_loss_value = compute_loss_confidence_band(
        6,
        model,
        min_epoch_size,
        train_dataset,
        batch_size,
        verbose=False,
        bottom_half=True,
    )

    if restart_model_full_minimization:
        batch_X, batch_y = train_dataset.get_batch(batch_size)
        model.initialize_model(batch_X.shape[1])

    curr_epoch_index = 0
    total_num_steps = 0

    while True:
        model = train_model(
            model,
            curr_epoch_size,
            train_dataset,
            batch_size,
            verbose=False,
            restart_model_full_minimization=False,
        )
        train_batch = train_dataset.get_batch(1000)
        with torch.no_grad():
            curr_loss = model.get_loss(train_batch[0], train_batch[1])
            curr_loss = curr_loss.detach()

        if verbose:
            print(
                "Curr loss ",
                curr_loss,
                "prev loss ",
                prev_loss_value,
                " epoch ",
                curr_epoch_index,
                " total num steps ",
                total_num_steps,
                " min loss value ",
                min_loss_value,
            )

        if (
            min_loss_value - eps < curr_loss and curr_loss < min_loss_value + eps
        ):  # eps and prev_loss_value - eps < curr_loss:
            # print(
            # "asdlfkmasdlfkmasdlkfmasldkfm", " Curr loss ", curr_loss, "Prev loss ",
            # prev_loss_value)
            return model

        min_loss_value = min(curr_loss.detach(), min_loss_value)
        prev_loss_value = curr_loss.detach()

        total_num_steps += curr_epoch_size

        curr_epoch_size = 2 * curr_epoch_size

        curr_epoch_index += 1

        if curr_epoch_index % eps_epoch_cycle == 0:
            eps *= 5
            max_epochs += 1
            print("Minimization Expanded max epochs and expanded eps ")

        if curr_epoch_index % max_epochs == 0:
            print(
                "Curr epoch index ",
                curr_epoch_index,
                "total num steps",
                total_num_steps,
            )
            curr_epoch_size = min_epoch_size
            # curr_epoch_index = 0
            prev_loss_value = float("inf")
            batch_X, batch_y = train_dataset.get_batch(batch_size)
            model.initialize_model(batch_X.shape[1])
            with torch.no_grad():
                curr_loss = model.get_loss(train_batch[0], train_batch[1])
                curr_loss = curr_loss.detach()
            print("Curr loss after restart ", curr_loss)


def train_model_counterfactual(
    model,
    num_steps,
    train_dataset,
    batch_size,
    query_batch,
    counterfactual_regularizer=1,
    verbose=False,
    restart_model_full_minimization=True,
):

    for i in range(num_steps):
        if verbose:
            print("train model iteration ", i)
        batch_X, batch_y = train_dataset.get_batch(batch_size)

        if i == 0:
            # model.initialize_model(batch_X)
            if restart_model_full_minimization:
                model.initialize_model(batch_X.shape[1])
            optimizer = torch.optim.Adam(model.network.parameters(), lr=0.01)

        optimizer.zero_grad()
        loss = model.get_loss(
            batch_X, batch_y
        ) - counterfactual_regularizer * torch.mean(model.predict_prob(query_batch))

        loss.backward()
        optimizer.step()

    return model


def train_model_counterfactual_with_stopping(
    model,
    loss_initial,
    loss_confidence_band,
    epoch_size,
    train_dataset,
    query_batch,
    batch_size,
    counterfactual_reg,
    verbose=False,
    restart_model_full_minimization=False,
    max_epochs=6,
    eps_epoch_cycle=30,
):
    loss_final = float("inf")
    all_data_X, all_data_Y = train_dataset.get_batch(10000000000)

    curr_epoch_index = 0
    initial_counterfactual_reg = counterfactual_reg
    initial_epoch_size = epoch_size

    while loss_final > loss_initial + loss_confidence_band:

        # print("Recomputing .... ")
        print(
            "Start training of conterfactual model",
            "loss initial ",
            loss_initial,
            " loss confidence band ",
            loss_confidence_band,
            " loss_final ",
            loss_final,
        )
        model = train_model_counterfactual(
            model,
            epoch_size,
            train_dataset,
            batch_size,
            query_batch,
            counterfactual_regularizer=counterfactual_reg,
            verbose=False,
            restart_model_full_minimization=False,
        )
        gc.collect()

        counterfactual_reg *= 0.5 * counterfactual_reg
        curr_epoch_index += 1
        epoch_size *= 2
        print("Counterfactual epoch ", curr_epoch_index)

        if curr_epoch_index % eps_epoch_cycle == 0:
            loss_confidence_band *= 5
            max_epochs += 1
            print(
                "Counterfactual expanded max epochs "
                "and expanded the loss confidence band "
            )

        if curr_epoch_index % max_epochs == 0:
            counterfactual_reg = initial_counterfactual_reg
            epoch_size = initial_epoch_size
            model.initialize_model(query_batch.shape[1])

        # EVALUATE THE EXPECTED LOSS
        with torch.no_grad():
            loss_final = model.get_loss(all_data_X, all_data_Y)

    print(
        "Start training of conterfactual model",
        "loss initial ",
        loss_initial,
        " loss confidence band ",
        loss_confidence_band,
        " loss_final ",
        loss_final,
    )

    return model


def compute_loss_confidence_band_with_stopping(
    num_loss_samples,
    model,
    min_epoch_size,
    train_dataset,
    batch_size,
    bottom_half=False,
    eps=0.0001,
):
    loss_values = []
    for i in range(num_loss_samples):
        model = train_model_with_stopping(
            model,
            min_epoch_size,
            train_dataset,
            batch_size,
            verbose=True,
            restart_model_full_minimization=True,
            eps=eps,
            max_epochs=6,
        )
        # model = train_model(
        # model, num_steps, train_dataset, batch_size, verbose = False,
        # restart_model_full_minimization = True
        # )
        all_data_X, all_data_Y = train_dataset.get_batch(10000000000)
        with torch.no_grad():
            loss = model.get_loss(all_data_X, all_data_Y)
            loss_values.append(loss.detach())

    # IPython.embed()
    # raise ValueError("asdflkm")
    if bottom_half:
        loss_values.sort()
        loss_values = loss_values[0 : int(len(loss_values) / 2)]
    return np.std(loss_values), np.mean(loss_values)


def compute_loss_confidence_band(
    num_loss_samples,
    model,
    num_steps,
    train_dataset,
    batch_size,
    verbose=False,
    bottom_half=False,
):
    loss_values = []
    for i in range(num_loss_samples):
        # print("loss sample ", i)
        model = train_model(
            model,
            num_steps,
            train_dataset,
            batch_size,
            verbose=False,
            restart_model_full_minimization=True,
        )
        all_data_X, all_data_Y = train_dataset.get_batch(10000000000)
        with torch.no_grad():
            loss = model.get_loss(all_data_X, all_data_Y)
            loss_values.append(loss.detach())

    if bottom_half:
        loss_values.sort()
        loss_values = loss_values[0 : int(len(loss_values) / 2)]

    return np.std(loss_values), np.mean(loss_values)


def gradient_step(model, optimizer, batch_X, batch_y):

    optimizer.zero_grad()
    loss = model.get_loss(batch_X, batch_y)
    loss.backward()
    optimizer.step()

    return model, optimizer


def run_regret_experiment_pytorch(
    dataset,
    training_mode,
    nn_params,
    linear_model_hparams,
    exploration_hparams,
    logging_frequency=100,
    num_experiments=5,
):
    # TODO: remove/pull up into hparams
    regret_wrt_baseline = True
    MLP = True
    num_full_minimization_steps = 200
    verbose = True
    restart_model_full_minimization = True
    estimate_loss_confidence_band = True

    (
        protected_datasets_train,
        protected_datasets_test,
        train_dataset,
        test_dataset,
    ) = get_dataset(dataset, nn_params.batch_size, 1000)
    baseline_model = TorchBinaryLogisticRegression(
        random_init=nn_params.random_init,
        fit_intercept=linear_model_hparams.fit_intercept,
        alpha=exploration_hparams.alpha,
        MLP=MLP,
        representation_layer_size=nn_params.representation_layer_size,
    )

    if (
        exploration_hparams.decision_type == "counterfactual"
        and exploration_hparams.epsilon_greedy
    ):
        raise ValueError(
            "Decision type set to counterfactual and epsilon greedy set to True"
        )
    if (
        exploration_hparams.decision_type == "counterfactual"
        and exploration_hparams.adjust_mahalanobis
    ):
        raise ValueError(
            "Decision type set to counterfactual and adjust_mahalanobis set to True"
        )

    if dataset == "MNIST":
        baseline_batch_size = nn_params.batch_size
    # TODO
    else:
        baseline_batch_size = 10

    baseline_model = train_model(
        baseline_model, nn_params.baseline_steps, train_dataset, baseline_batch_size
    )
    print("Finished training baseline model")

    with torch.no_grad():
        baseline_batch_test, protected_batches_test = get_batches(
            protected_datasets_test, test_dataset, 1000
        )
        baseline_accuracy, protected_accuracies = get_accuracies(
            baseline_batch_test,
            protected_batches_test,
            baseline_model,
            linear_model_hparams.threshold,
        )
        loss_validation_baseline = baseline_model.get_loss(
            baseline_batch_test[0], baseline_batch_test[1]
        )

    print("Baseline model accuracy {}".format(baseline_accuracy))

    # num_protected_groups = len(protected_datasets_train)
    # wass_distances = [[] for _ in range(num_protected_groups)]
    # logging_counters = [[] for _ in range(num_protected_groups)]
    # protected_accuracies_list = [[] for _ in range(num_protected_groups)]
    # biased_protected_accuracies_list = [[] for _ in range(num_protected_groups)]
    # losses = []
    # accuracies = []
    accuracies_list = []
    biased_accuracies_list = []

    loss_validation = []
    loss_validation_biased = []

    train_regret = []

    counter = 0
    biased_data_totals = 0

    model = TorchBinaryLogisticRegression(
        random_init=nn_params.random_init,
        fit_intercept=linear_model_hparams.fit_intercept,
        alpha=exploration_hparams.alpha,
        MLP=MLP,
        representation_layer_size=nn_params.representation_layer_size,
    )
    model_biased = TorchBinaryLogisticRegression(
        random_init=nn_params.random_init,
        fit_intercept=linear_model_hparams.fit_intercept,
        alpha=exploration_hparams.alpha,
        MLP=MLP,
        representation_layer_size=nn_params.representation_layer_size,
    )

    if exploration_hparams.decision_type == "counterfactual":
        model_biased_prediction = TorchBinaryLogisticRegression(
            random_init=nn_params.random_init,
            fit_intercept=linear_model_hparams.fit_intercept,
            alpha=exploration_hparams.alpha,
            MLP=MLP,
            representation_layer_size=nn_params.representation_layer_size,
        )

    cummulative_data_covariance = []
    inverse_cummulative_data_covariance = []

    train_accuracies_biased = []
    timesteps = []

    if training_mode == "full_minimization":
        biased_dataset = GrowingNumpyDataSet()
        unbiased_dataset = GrowingNumpyDataSet()

    while counter < nn_params.max_num_steps:
        counter += 1

        global_batch, protected_batches = get_batches(
            protected_datasets_train, train_dataset, nn_params.batch_size
        )
        batch_X, batch_y = global_batch

        if counter == 1:
            # model.initialize_model(batch_X)
            model.initialize_model(batch_X.shape[1])
            # model_biased.initialize_model(batch_X)
            model_biased.initialize_model(batch_X.shape[1])
            if exploration_hparams.decision_type == "counterfactual":
                model_biased_prediction.initialize_model(batch_X.shape[1])
                # model_biased_prediction.initialize_model(batch_X)

            optimizer_model = torch.optim.Adam(model.network.parameters(), lr=0.01)
            optimizer_biased = torch.optim.Adam(
                model_biased.network.parameters(), lr=0.01
            )

        # TRAIN THE UNBIASED MODEL
        if training_mode == "full_minimization":
            unbiased_dataset.add_data(batch_X, batch_y)
            # model = train_model(
            # model, num_full_minimization_steps, unbiased_dataset, batch_size,
            # restart_model_full_minimization = restart_model_full_minimization
            # )
            print(
                "Start of full minimization training of the unbiased model -- iteration ",
                counter,
            )
            model = train_model_with_stopping(
                model,
                num_full_minimization_steps,
                unbiased_dataset,
                nn_params.batch_size,
                verbose=True,
                restart_model_full_minimization=restart_model_full_minimization,
                eps=0.0001 * np.log(counter + 2) / 2,
            )

            gc.collect()

        elif training_mode == "gradient_step":
            model, optimizer_model = gradient_step(
                model, optimizer_model, batch_X, batch_y
            )

        if exploration_hparams.decision_type == "simple":
            # Training biased model
            global_biased_prediction, protected_biased_predictions = get_predictions(
                global_batch,
                protected_batches,
                model_biased,
                inverse_cummulative_data_covariance,
            )

        elif exploration_hparams.decision_type == "counterfactual":
            print(f"Training mode: {training_mode}")
            if training_mode != "full_minimization":
                raise ValueError(
                    "The counterfactual decision mode is incompatible with all "
                    "training modes different from full_minimization"
                )

            if biased_dataset.get_size() == 0:
                # ACCEPT ALL POINTS IF THE BIASED DATASET IS NOT INITIALIZED
                global_biased_prediction = [1 for _ in range(nn_params.batch_size)]
            # Evaluate the loss over the existing dataset.
            else:
                # EVALUATE THE EXPECTED LOSS
                with torch.no_grad():
                    all_data_X, all_data_Y = biased_dataset.get_batch(10000000000)
                    loss_initial = model_biased.get_loss(all_data_X, all_data_Y)
                if estimate_loss_confidence_band:
                    print("Starting computation of the loss confidence band ")
                    (
                        loss_confidence_band,
                        mean_loss_confidence_band,
                    ) = compute_loss_confidence_band_with_stopping(
                        10,
                        model_biased,
                        num_full_minimization_steps,
                        biased_dataset,
                        nn_params.batch_size,
                        bottom_half=True,
                        eps=0.0001 * np.log(counter + 2) / 2,
                    )
                    loss_confidence_band *= 2
                    loss_confidence_band += 0.0001 * np.log(counter + 2) / 2
                    # loss_confidence_band = 2*compute_loss_confidence_band(
                    # 10, model_biased, num_full_minimization_steps, biased_dataset,
                    # batch_size, verbose = False
                    # )
                    gc.collect()
                else:
                    loss_confidence_band = loss_initial

                counterfactual_reg = 1  # COUNTERFACTUAL REGULARIZATION

                model_biased_prediction = train_model_counterfactual_with_stopping(
                    model_biased_prediction,
                    loss_initial,
                    loss_confidence_band,
                    num_full_minimization_steps,
                    biased_dataset,
                    batch_X,
                    nn_params.batch_size,
                    counterfactual_reg,
                    verbose=False,
                    restart_model_full_minimization=False,
                )

                # loss_final = float("inf")

                # while loss_final > loss_initial + loss_confidence_band:
                #   print("Recomputing .... ")
                #   print(
                #     "Start training of conterfactual model", "loss initial ",
                #     loss_initial, " loss confidence band ", loss_confidence_band,
                # " loss_final ", loss_final)
                #   model_biased_prediction = train_model_counterfactual(
                # model_biased_prediction,  num_full_minimization_steps, biased_dataset,
                # batch_size, batch_X,
                #     counterfactual_regularizer = counterfactual_reg, verbose = (
                # False, restart_model_full_minimization = False)
                #   gc.collect()

                #   ### EVALUATE THE EXPECTED LOSS ##
                #   with torch.no_grad():
                #     loss_final = model_biased_prediction.get_loss(
                # all_data_X, all_data_Y)

                #   counterfactual_reg *= .5*counterfactual_reg

                global_biased_prediction, _ = get_predictions(
                    global_batch, protected_batches, model_biased_prediction
                )
                print("Global biased prediction ", global_biased_prediction)

        biased_batch_X = []
        biased_batch_y = []
        biased_batch_size = 0
        biased_train_accuracy = 0
        batch_regret = 0

        for i in range(len(global_biased_prediction)):
            accept_point = global_biased_prediction[
                i
            ] > linear_model_hparams.biased_threshold or (
                exploration_hparams.epsilon_greedy
                and np.random.random() < exploration_hparams.epsilon
            )

            biased_train_accuracy += (accept_point == batch_y[i]) * 1.0
            if regret_wrt_baseline:
                batch_regret += baseline_accuracy - (accept_point == batch_y[i]) * 1.0

            else:
                if accept_point and batch_y[i] == 0:
                    batch_regret += 1
                elif not accept_point and batch_y[i] == 1:
                    batch_regret += 1.0

            if accept_point:
                biased_batch_X.append(batch_X[i])
                biased_batch_y.append(batch_y[i])
                biased_batch_size += 1
        biased_batch_X = np.array(biased_batch_X)
        biased_batch_y = np.array(biased_batch_y)

        biased_train_accuracy = biased_train_accuracy / len(global_biased_prediction)
        batch_regret = batch_regret / len(global_biased_prediction) * 1.0

        biased_data_totals += biased_batch_size

        # Train biased model on biased data
        if biased_batch_size > 0:
            if training_mode == "full_minimization":
                biased_dataset.add_data(biased_batch_X, biased_batch_y)
                # model_biased = train_model(
                #     model_biased, num_full_minimization_steps, biased_dataset,
                # batch_size, restart_model_full_minimization =
                # restart_model_full_minimization)
                model_biased = train_model_with_stopping(
                    model_biased,
                    num_full_minimization_steps,
                    biased_dataset,
                    nn_params.batch_size,
                    verbose=True,
                    restart_model_full_minimization=restart_model_full_minimization,
                    eps=0.0001 * np.log(counter + 2) / 2,
                )
                gc.collect()

            elif training_mode == "gradient_step":
                model_biased, optimizer_biased = gradient_step(
                    model_biased, optimizer_biased, biased_batch_X, biased_batch_y
                )

            else:
                raise ValueError("Unrecognized training mode")

            if exploration_hparams.decision_type == "simple":
                representation_X = model_biased.get_representation(
                    biased_batch_X
                ).detach()
                representation_X = representation_X.numpy()
                if exploration_hparams.adjust_mahalanobis:
                    if len(cummulative_data_covariance) == 0:
                        cummulative_data_covariance = np.dot(
                            np.transpose(representation_X), representation_X
                        )
                    else:
                        cummulative_data_covariance = (
                            exploration_hparams.mahalanobis_discount_factor
                            * cummulative_data_covariance
                            + np.dot(np.transpose(representation_X), representation_X)
                        )

                    # This can be done instead by using the Sherman-Morrison Formula.
                    inverse_cummulative_data_covariance = torch.from_numpy(
                        np.linalg.inv(
                            exploration_hparams.mahalanobis_regularizer
                            * np.eye(representation_X.shape[1])
                            + cummulative_data_covariance
                        )
                    ).float()

        # DIAGNOSTICS
        # Compute accuracy diagnostics
        if counter % logging_frequency * 1.0 == 0:
            train_regret.append(batch_regret)
            train_accuracies_biased.append(biased_train_accuracy)
            timesteps.append(counter)
            global_batch_test, protected_batches_test = get_batches(
                protected_datasets_test, test_dataset, 1000
            )
            batch_X_test, batch_y_test = global_batch_test
            global_probabilities_list, protected_predictions = get_predictions(
                global_batch_test, protected_batches_test, model
            )
            total_accuracy, protected_accuracies = get_accuracies(
                global_batch_test,
                protected_batches_test,
                model,
                linear_model_hparams.threshold,
            )

            with torch.no_grad():
                # Compute loss diagnostics
                biased_loss = model_biased.get_loss(batch_X_test, batch_y_test)
                loss = model.get_loss(batch_X_test, batch_y_test)
                loss_validation.append(loss.detach())
                loss_validation_biased.append(biased_loss.detach())

            accuracies_list.append(total_accuracy)
            (
                biased_global_probabilities_list,
                biased_protected_predictions,
            ) = get_predictions(global_batch_test, protected_batches_test, model_biased)
            biased_total_accuracy, biased_protected_accuracies = get_accuracies(
                global_batch_test,
                protected_batches_test,
                model_biased,
                linear_model_hparams.threshold,
            )
            biased_accuracies_list.append(biased_total_accuracy)

            # Compute training biased accuracy
            train_biased_batch = biased_dataset.get_batch(1000)
            biased_train_accuracy = get_accuracies_simple(
                train_biased_batch, model_biased, linear_model_hparams.threshold
            )
            with torch.no_grad():
                loss_train_biased = model_biased.get_loss(
                    train_biased_batch[0], train_biased_batch[1]
                )
                loss_train_biased = loss_train_biased.detach()

            if verbose:
                # IPython.embed()
                # raise ValueError("as;kdfm")
                print("Iteration {}".format(counter))
                print(
                    "Total proportion of biased data {}".format(
                        1.0 * biased_data_totals / (nn_params.batch_size * counter)
                    )
                )
                print("Biased TRAIN accuracy  ", biased_train_accuracy)
                print("Biased TRAIN loss ", loss_train_biased)

                print(
                    "                                                               Baseline accuracy ",
                    baseline_accuracy,
                )
                # Compute the global accuracy.
                print(
                    "                                                               Unbiased Accuracy ",
                    total_accuracy,
                )

                # Compute the global accuracy.
                print(
                    "                                                               Biased Accuracy ",
                    biased_total_accuracy,
                )

                print(
                    "                                                               Validation Loss Unbiased ",
                    loss_validation[-1],
                )

                print(
                    "                                                               Validation Loss Biased ",
                    loss_validation_biased[-1],
                )

            test_biased_accuracies_cum_averages = np.cumsum(biased_accuracies_list)
            test_biased_accuracies_cum_averages = (
                test_biased_accuracies_cum_averages / (np.arange(len(timesteps)) + 1)
            )
            accuracies_cum_averages = np.cumsum(accuracies_list)
            accuracies_cum_averages = accuracies_cum_averages / (
                np.arange(len(timesteps)) + 1
            )
            train_biased_accuracies_cum_averages = np.cumsum(train_accuracies_biased)
            train_biased_accuracies_cum_averages = (
                train_biased_accuracies_cum_averages / (np.arange(len(timesteps)) + 1)
            )
            train_cum_regret = np.cumsum(train_regret)

    return (
        timesteps,
        test_biased_accuracies_cum_averages,
        accuracies_cum_averages,
        train_biased_accuracies_cum_averages,
        train_cum_regret,
        loss_validation,
        loss_validation_biased,
        loss_validation_baseline,
        baseline_accuracy,
    )
