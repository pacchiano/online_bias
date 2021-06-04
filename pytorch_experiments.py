import numpy as np
import matplotlib.pyplot as plt

import os
import pickle
import ray
import sys
import time

from dataclasses import dataclass
from experiment_regret import run_regret_experiment_pytorch
from typing import Any

LINEWIDTH = 3.5
LINESTYLE = "dashed"
STD_GAP = 0.5
ALPHA = 0.1


def algo_to_params(algo):
    exploration_hparams = ExplorationHparams()
    exploration_hparams.epsilon = 0.2
    if algo == "Eps_Greedy":
        exploration_hparams.decision_type = "simple"
        exploration_hparams.epsilon_greedy = True
    elif algo == "Greedy":
        exploration_hparams.decision_type = "simple"
        exploration_hparams.epsilon_greedy = False
    elif algo == "NeuralUCB":
        exploration_hparams.decision_type = "simple"
        exploration_hparams.adjust_mahalanobis = True
        exploration_hparams.mahalanobis_discount_factor = 0.9
        exploration_hparams.alpha = 4
    elif algo == "PLOT":
        exploration_hparams.decision_type = "counterfactual"
    else:
        raise ValueError("Unsupported Online Algorithm")
    return exploration_hparams


@dataclass
class NNParams:
    representation_layer_size = 40
    max_num_steps = 100
    baseline_steps = 10000
    batch_size = 32
    num_full_minimization_steps = 100
    pseudo_steps_multiplier = 8
    random_init = True
    restart_model_full_minimization = True
    weight_decay = 0.0
    pseudolabel = True


@dataclass
class LinearModelHparams:
    logistic_learning_rate = 0.01
    threshold = 0.5
    biased_threshold = 0.5
    fit_intercept = True


@dataclass
class ExplorationHparams:
    mahalanobis_discount_factor = 1
    mahalanobis_regularizer = 0.1
    epsilon = 0.2
    alpha = 1
    decision_type = "counterfactual"
    epsilon_greedy = False
    adjust_mahalanobis = False
    loss_confidence_band = None
    regret_wrt_baseline = True


@dataclass
class ExperimentResults:
    mean_accuracies_cum_averages: Any
    std_accuracies_cum_averages: Any
    mean_train_biased_accuracies_cum_averages: Any
    std_train_biased_accuracies_cum_averages: Any
    mean_test_biased_accuracies_cum_averages: Any
    std_test_biased_accuracies_cum_averages: Any
    mean_train_cum_regret_averages: Any
    std_train_cum_regret_averages: Any
    mean_loss_validation_averages: Any
    std_loss_validation_averages: Any
    mean_loss_validation_biased_averages: Any
    std_loss_validation_biased_averages: Any
    mean_loss_validation_baseline_summary: Any
    std_loss_validation_baseline_summary: Any
    mean_accuracy_validation_baseline_summary: Any
    std_accuracy_validation_baseline_summary: Any


def run_experiment_parallel(
    dataset,
    training_mode,
    nn_params,
    linear_model_hparams,
    exploration_hparams,
    logging_frequency,
):
    (
        timesteps,
        test_biased_accuracies_cum_averages,
        accuracies_cum_averages,
        train_biased_accuracies_cum_averages,
        train_cum_regret,
        loss_validation,
        loss_validation_biased,
        loss_baseline,
        baseline_accuracy,
        train_error_breakdown,
        test_error_breakdown,
        pseudo_error_breakdown,
        eps_error_breakdown
    ) = run_regret_experiment_pytorch(
        dataset,
        training_mode,
        nn_params,
        linear_model_hparams,
        exploration_hparams,
        logging_frequency,
    )
    return (
        timesteps,
        test_biased_accuracies_cum_averages,
        accuracies_cum_averages,
        train_biased_accuracies_cum_averages,
        train_cum_regret,
        loss_validation,
        loss_validation_biased,
        loss_baseline,
        baseline_accuracy,
        train_error_breakdown,
        test_error_breakdown,
        pseudo_error_breakdown,
        eps_error_breakdown
    )


def configure_directories(dataset, nn_params, linear, algo):
    path = os.getcwd()
    network_type = (
        "Linear{}".format(nn_params.representation_layer_size) if linear else "MLP"
    )
    base_data_directory = "{}/experiment_results/{}/{}/data".format(
        path, dataset, algo
    )
    base_figs_directory = "{}/experiment_results/{}/{}/figs".format(
        path, dataset, algo
    )

    if not os.path.isdir(base_data_directory):
        try:
            os.makedirs(base_data_directory)
        except OSError:
            print("Creation of data directories failed")
        else:
            print("Successfully created the data directory")

    if not os.path.isdir(base_figs_directory):
        try:
            os.makedirs(base_figs_directory)
        except OSError:
            print("Creation of figs directories failed")
        else:
            print("Successfully created the figs directory")
    return network_type, base_figs_directory, base_data_directory


def run_experiments(
    dataset,
    training_mode,
    nn_params,
    linear_model_hparams,
    exploration_hparams,
    logging_frequency,
    num_experiments,
    use_ray,
):
    if use_ray:
        experiment_summaries = [
            run_experiment_parallel.remote(
                dataset,
                training_mode,
                nn_params,
                linear_model_hparams,
                exploration_hparams,
                logging_frequency,
            )
            for _ in range(num_experiments)
        ]
        experiment_summaries = ray.get(experiment_summaries)

    else:
        experiment_summaries = [
            run_experiment_parallel(
                dataset,
                training_mode,
                nn_params,
                linear_model_hparams,
                exploration_hparams,
                logging_frequency,
            )
            for _ in range(num_experiments)
        ]
    return experiment_summaries


def analyze_experiments(
    experiment_summaries,
    dataset,
    training_mode,
    nn_params,
    linear_model_hparams,
    exploration_hparams,
    logging_frequency,
    num_experiments,
):
    test_biased_accuracies_cum_averages_summary = np.zeros(
        (num_experiments, int(nn_params.max_num_steps / logging_frequency))
    )
    accuracies_cum_averages_summary = np.zeros(
        (num_experiments, int(nn_params.max_num_steps / logging_frequency))
    )
    train_biased_accuracies_cum_averages_summary = np.zeros(
        (num_experiments, int(nn_params.max_num_steps / logging_frequency))
    )
    train_cum_regret_summary = np.zeros(
        (num_experiments, int(nn_params.max_num_steps / logging_frequency))
    )
    loss_validation_summary = np.zeros(
        (num_experiments, int(nn_params.max_num_steps / logging_frequency))
    )
    loss_validation_biased_summary = np.zeros(
        (num_experiments, int(nn_params.max_num_steps / logging_frequency))
    )

    loss_validation_baseline_summary = np.zeros(num_experiments)
    accuracy_validation_baseline_summary = np.zeros(num_experiments)

    for j in range(num_experiments):
        test_biased_accuracies_cum_averages_summary[j, :] = experiment_summaries[j][1]
        accuracies_cum_averages_summary[j, :] = experiment_summaries[j][2]
        train_biased_accuracies_cum_averages_summary[j, :] = experiment_summaries[j][3]
        train_cum_regret_summary[j, :] = experiment_summaries[j][4]
        loss_validation_summary[j, :] = experiment_summaries[j][5]
        loss_validation_biased_summary[j, :] = experiment_summaries[j][6]
        loss_validation_baseline_summary[j] = experiment_summaries[j][7]
        accuracy_validation_baseline_summary[j] = experiment_summaries[j][8]

    mean_test_biased_accuracies_cum_averages = np.mean(
        test_biased_accuracies_cum_averages_summary, axis=0
    )
    std_test_biased_accuracies_cum_averages = np.std(
        test_biased_accuracies_cum_averages_summary, axis=0
    )

    mean_accuracies_cum_averages = np.mean(accuracies_cum_averages_summary, axis=0)
    std_accuracies_cum_averages = np.std(accuracies_cum_averages_summary, axis=0)

    mean_train_biased_accuracies_cum_averages = np.mean(
        train_biased_accuracies_cum_averages_summary, axis=0
    )
    std_train_biased_accuracies_cum_averages = np.std(
        train_biased_accuracies_cum_averages_summary, axis=0
    )

    mean_train_cum_regret_averages = np.mean(train_cum_regret_summary, axis=0)
    std_train_cum_regret_averages = np.std(train_cum_regret_summary, axis=0)

    mean_loss_validation_averages = np.mean(loss_validation_summary, axis=0)
    std_loss_validation_averages = np.std(loss_validation_summary, axis=0)

    mean_loss_validation_biased_averages = np.mean(
        loss_validation_biased_summary, axis=0
    )
    std_loss_validation_biased_averages = np.std(loss_validation_biased_summary, axis=0)
    mean_loss_validation_baseline_summary = np.mean(loss_validation_baseline_summary)
    std_loss_validation_baseline_summary = np.std(loss_validation_baseline_summary)

    mean_accuracy_validation_baseline_summary = np.mean(
        accuracy_validation_baseline_summary
    )
    std_accuracy_validation_baseline_summary = np.std(
        accuracy_validation_baseline_summary
    )
    return ExperimentResults(
        mean_accuracies_cum_averages,
        std_accuracies_cum_averages,
        mean_train_biased_accuracies_cum_averages,
        std_train_biased_accuracies_cum_averages,
        mean_test_biased_accuracies_cum_averages,
        std_test_biased_accuracies_cum_averages,
        mean_train_cum_regret_averages,
        std_train_cum_regret_averages,
        mean_loss_validation_averages,
        std_loss_validation_averages,
        mean_loss_validation_biased_averages,
        std_loss_validation_biased_averages,
        mean_loss_validation_baseline_summary,
        std_loss_validation_baseline_summary,
        mean_accuracy_validation_baseline_summary,
        std_accuracy_validation_baseline_summary,
    )


def plot_helper(timesteps, accuracies, accuracies_stds, label, color, broadcast=False):
    if broadcast:
        accuracies = np.array([accuracies] * len(timesteps))
        accuracies_stds = np.array([accuracies_stds] * len(timesteps))
    plt.plot(
        timesteps,
        accuracies,
        label=label,
        linestyle=LINESTYLE,
        linewidth=LINEWIDTH,
        color=color,
    )
    plt.fill_between(
        timesteps,
        accuracies - STD_GAP * accuracies_stds,
        accuracies + STD_GAP * accuracies_stds,
        color=color,
        alpha=ALPHA,
    )


def plot_title(
    plot_type,
    dataset,
    network_type,
    training_mode,
    exploration_hparams,
):
    if plot_type == "accuracy":
        plot_type_prefix = "Test and Train Accuracies"
        plot_type_file_prefix = "test_train_accuracies"
    elif plot_type == "regret":
        plot_type_prefix = "Regret"
        plot_type_file_prefix = "regret"
    elif plot_type == "loss":
        plot_type_prefix = "Loss"
        plot_type_file_prefix = "loss"

    if exploration_hparams.decision_type == "simple":
        if exploration_hparams.epsilon_greedy:
            plt.title(
                (
                    f"{plot_type_prefix} {dataset} - "
                    f"Epsilon Greedy {exploration_hparams.epsilon} - {network_type} - {training_mode}"
                ),
                fontsize=8,
            )
            plot_name = "{}_{}_epsgreedy_{}_{}_{}".format(
                dataset,
                plot_type_file_prefix,
                exploration_hparams.epsilon,
                network_type,
                training_mode,
            )
        if exploration_hparams.adjust_mahalanobis:
            plt.title(
                (
                    f"{plot_type_prefix} {dataset} - Optimism alpha {exploration_hparams.alpha} "
                    f"- Mreg {exploration_hparams.mahalanobis_regularizer} "
                    f"- Mdisc {exploration_hparams.mahalanobis_discount_factor} - "
                    f"{network_type} - {training_mode}"
                ),
                fontsize=8,
            )
            plot_name = "{}_{}_optimism_alpha_{}_mahreg_{}_mdisc_{}_{}_{}".format(
                dataset,
                plot_type_file_prefix,
                exploration_hparams.alpha,
                exploration_hparams.mahalanobis_regularizer,
                exploration_hparams.mahalanobis_discount_factor,
                network_type,
                training_mode,
            )
        if (
            not exploration_hparams.epsilon_greedy and not exploration_hparams.adjust_mahalanobis
        ):
            plt.title(
                "{} {} - {} - {} ".format(
                    plot_type_prefix, dataset, network_type, training_mode
                ),
                fontsize=8,
            )
            plot_name = "{}_{}_biased_{}_{}".format(
                dataset, plot_type_file_prefix, network_type, training_mode
            )
    elif exploration_hparams.decision_type == "counterfactual":
        plt.title(
            "{} {} - {} - {} - {}".format(
                plot_type_prefix,
                dataset,
                network_type,
                training_mode,
                exploration_hparams.decision_type,
            ),
            fontsize=8,
        )
        plot_name = "{}_{}_biased_{}_{}_{}".format(
            dataset,
            plot_type_file_prefix,
            network_type,
            training_mode,
            exploration_hparams.decision_type,
        )

    else:
        raise ValueError(
            "Decision type not recognized {}".format(exploration_hparams.decision_type)
        )
    return plot_name


def plot_results(
    timesteps,
    experiment_results,
    network_type,
    base_figs_directory,
    dataset,
    training_mode,
    exploration_hparams,
):
    # ACCURACY PLOTS
    plot_helper(
        timesteps,
        experiment_results.mean_test_biased_accuracies_cum_averages,
        experiment_results.std_test_biased_accuracies_cum_averages,
        "Biased Model Test - no decision adjustment",
        "blue",
    )
    plot_helper(
        timesteps,
        experiment_results.mean_accuracies_cum_averages,
        experiment_results.std_accuracies_cum_averages,
        label="Unbiased Model Test - all data train",
        color="red",
    )
    plot_helper(
        timesteps,
        experiment_results.mean_train_biased_accuracies_cum_averages,
        experiment_results.std_train_biased_accuracies_cum_averages,
        label="Online Biased Model - filtered data train",
        color="violet",
    )
    plot_helper(
        timesteps,
        experiment_results.mean_accuracy_validation_baseline_summary,
        experiment_results.std_accuracy_validation_baseline_summary,
        label="Baseline Accuracy",
        color="black",
        broadcast=True
    )

    plot_name = plot_title(
        "accuracy",
        dataset,
        network_type,
        training_mode,
        exploration_hparams,
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Accuracy")
    lg = plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8, loc="upper left")
    print(f"Saving plot to {base_figs_directory}/{plot_name}.png")
    plt.savefig(
        "{}/{}.png".format(base_figs_directory, plot_name),
        bbox_extra_artists=(lg,),
        bbox_inches="tight",
    )
    plt.close("all")

    # REGRET PLOTS
    plot_helper(
        timesteps,
        experiment_results.mean_train_cum_regret_averages,
        experiment_results.std_train_cum_regret_averages,
        label="Regret",
        color="blue",
    )
    plot_name = plot_title(
        "regret",
        dataset,
        network_type,
        training_mode,
        exploration_hparams,
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Regret")
    lg = plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8, loc="upper left")
    plt.savefig(
        "{}/{}.png".format(base_figs_directory, plot_name),
        bbox_extra_artists=(lg,),
        bbox_inches="tight",
    )
    plt.close("all")

    # LOSS PLOTS
    plot_helper(
        timesteps,
        experiment_results.mean_loss_validation_averages,
        experiment_results.std_loss_validation_averages,
        label="Unbiased model loss",
        color="blue",
    )
    plot_helper(
        timesteps,
        experiment_results.mean_loss_validation_biased_averages,
        experiment_results.std_loss_validation_biased_averages,
        label="Biased model loss",
        color="red",
    )
    plot_helper(
        timesteps,
        experiment_results.mean_loss_validation_baseline_summary,
        experiment_results.std_loss_validation_baseline_summary,
        label="Baseline Loss",
        color="black",
        broadcast=True
    )
    plot_name = plot_title(
        "loss",
        dataset,
        network_type,
        training_mode,
        exploration_hparams,
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Loss")
    lg = plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8, loc="upper left")

    plt.savefig(
        "{}/{}.png".format(base_figs_directory, plot_name),
        bbox_extra_artists=(lg,),
        bbox_inches="tight",
    )
    plt.close("all")


def run_and_plot(
    dataset,
    training_mode,
    nn_params,
    linear_model_hparams,
    exploration_hparams,
    logging_frequency,
    num_experiments,
    use_ray,
    algo,
):
    if use_ray:
        ray.init()
    start_time = time.time()
    linear = False
    network_type, base_figs_directory, base_data_directory = configure_directories(
        dataset, nn_params, linear, algo
    )
    if exploration_hparams.epsilon_greedy and exploration_hparams.adjust_mahalanobis:
        raise ValueError("Both epsilon greedy and adjust mahalanobis are on!")

    print(
        f"Starting Experiment {dataset} T{nn_params.max_num_steps} "
        f"{training_mode} {repr(exploration_hparams)}"
    )
    experiment_summaries = run_experiments(
        dataset,
        training_mode,
        nn_params,
        linear_model_hparams,
        exploration_hparams,
        logging_frequency,
        num_experiments,
        use_ray,
    )
    experiment_results = analyze_experiments(
        experiment_summaries,
        dataset,
        training_mode,
        nn_params,
        linear_model_hparams,
        exploration_hparams,
        logging_frequency=logging_frequency,
        num_experiments=num_experiments,
    )
    timesteps = experiment_summaries[0][0]
    plot_results(
        timesteps,
        experiment_results,
        network_type,
        base_figs_directory,
        dataset,
        training_mode,
        exploration_hparams,
    )
    pickle.dump(
        (
            timesteps,
            nn_params.max_num_steps,
            experiment_results,
        ),
        open("{}/{}.p".format(base_data_directory, "data_dump"), "wb"),
    )
    train_breakdowns = []
    test_breakdowns = []
    pseudo_breakdowns = []
    eps_breakdowns = []
    for summary in experiment_summaries:
        train_breakdowns.append(summary[-4])
        test_breakdowns.append(summary[-3])
        pseudo_breakdowns.append(summary[-2])
        eps_breakdowns.append(summary[-1])
    # print("Train Breakdowns")
    # print(train_breakdowns)
    # print("Test Breakdowns")
    # print(test_breakdowns)
    # print("Pseudo Breakdowns")
    # print(pseudo_breakdowns)
    # print("Eps Breakdowns")
    # print(eps_breakdowns)
    pickle.dump(
        # FPR/FNR
        (
            train_breakdowns,
            test_breakdowns,
            pseudo_breakdowns,
            eps_breakdowns
        ),
        open("{}/{}.p".format(base_data_directory, "fnr_dump"), "wb"),
    )
    end_time = time.time()
    total = end_time - start_time
    print(f"Total runtime: {total}")


if __name__ == "__main__":
    T = 2000
    BASELINE_STEPS = 20_000
    BATCH_SIZE = 32

    nn_params = NNParams()
    nn_params.max_num_steps = T
    nn_params.batch_size = BATCH_SIZE
    nn_params.baseline_steps = BASELINE_STEPS
    linear_model_hparams = LinearModelHparams()
    exploration_hparams = ExplorationHparams()
    logging_frequency = min(10, T % 5)
    training_mode = "full_minimization"
    MULTI = False
    RAY = False
    NUM_EXPERIMENTS = 1
    try:
        MULTI = sys.argv[1]
    except IndexError:
        pass
    try:
        RAY = sys.argv[2]
    except IndexError:
        pass
    try:
        NUM_EXPERIMENTS = sys.argv[3]
    except IndexError:
        pass

    if RAY:
        NUM_EXPERIMENTS = 5
        # run_experiment_parallel = ray.remote(num_gpus=1)(run_experiment_parallel)
        run_experiment_parallel = ray.remote(run_experiment_parallel)
        RAY = True

    if MULTI:
        algos = ["Eps_Greedy", "Greedy", "NeuralUCB", "PLOT"]
        datasets = ["Adult", "Bank", "MultiSVM", "MNIST"]
        for algo_name in algos:
            exploration_hparams = algo_to_params(algo_name)
            for dataset in datasets:
                run_and_plot(
                    dataset,
                    training_mode,
                    nn_params,
                    linear_model_hparams,
                    exploration_hparams,
                    logging_frequency,
                    NUM_EXPERIMENTS,
                    RAY,
                    algo_name
                )
    else:
        dataset = "Adult"
        algo_name = "PLOT"
        exploration_hparams = algo_to_params(algo_name)
        run_and_plot(
            dataset,
            training_mode,
            nn_params,
            linear_model_hparams,
            exploration_hparams,
            logging_frequency,
            NUM_EXPERIMENTS,
            RAY,
            algo_name
        )
