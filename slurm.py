#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pathlib
import shutil

import submitit

from pytorch_experiments import (
    run_and_plot,
    ExplorationHparams,
    LinearModelHparams,
    NNParams,
)

PARALLEL = True
FAST = True
# FAST = False
# VERSION = "_100t_fpr_norm_zero"
VERSION = "ray_test"

JOB_PREFIX = "fair_bandits_test"
PARALLEL_STR = "_parallel" if PARALLEL else ""
JOB_NAME = f"{JOB_PREFIX}{PARALLEL_STR}_{VERSION}"


def copy_and_run_with_config(
    run_fn, run_config, directory, parallel=False, **cluster_config,
):
    print("Let's use slurm!")
    working_directory = pathlib.Path(directory) / cluster_config["job_name"]
    ignore_list = [
        "lightning_logs",
        "logs",
        "checkpoints",
        "experiments",
        "experiment_results",
        ".git",
        "output",
        "val.csv",
        "train.csv",
    ]
    shutil.copytree(".", working_directory, ignore=lambda x, y: ignore_list)
    os.chdir(working_directory)
    print(f"Running at {working_directory}")

    executor = submitit.SlurmExecutor(folder=working_directory)
    executor.update_parameters(**cluster_config)
    if parallel:
        jobs = executor.map_array(
            run_fn,
            *args,
            # [run_config["dataset"]]*10,
            # [run_config["training_mode"]]*10,
            # [run_config["nn_params"]]*10,
            # [run_config["linear_model_hparams"]]*10,
            # [run_config["exploration_hparams"]]*10,
            # [run_config["logging_frequency"]]*10,
            # [run_config["num_experiments"]]*10
        )
        print(f"job_ids: {jobs}")
    else:
        job = executor.submit(run_fn, run_config)
        print(f"job_id: {job}")


def get_parallel_args():
    datasets = ["MultiSVM", "Adult", "MNIST"]
    training_modes = ["full_minimization"] * len(datasets)
    nn_params = NNParams()
    # Fairly fast, decent capacity.
    # [10, 40, 100]
    nn_params.representation_layer_size = 40
    nn_param_list = [nn_params] * 3
    for nn_param in nn_param_list:
        # [30, 100, 200]
        # nn_param.max_num_steps = 100
        nn_param.max_num_steps = 30
    linear_model_hparams = [LinearModelHparams()] * len(datasets)
    exploration_hparams = [ExplorationHparams()] * len(datasets)
    for eh in exploration_hparams:
        eh.loss_confidence_band = 0
    # TODO: work with Ray
    num_experiments = [5] * len(datasets)
    logging_frequency = [10] * len(datasets)
    return [
        datasets, training_modes, nn_param_list, linear_model_hparams,
        exploration_hparams, num_experiments, logging_frequency
    ]


def get_args():
    dataset = "MultiSVM"
    nn_params = NNParams()
    nn_params.max_num_steps = 200
    linear_model_hparams = LinearModelHparams()
    exploration_hparams = ExplorationHparams()
    logging_frequency = 10
    # TODO: work with Ray
    num_experiments = 1

    if FAST:
        nn_params.max_num_steps = 2
        nn_params.baseline_steps = 10
        training_mode = "gradient_step"
        exploration_hparams.decision_type = "simple"
    return {
        "dataset": dataset,
        "training_mode": training_mode,
        "nn_params": nn_params,
        "linear_model_hparams": linear_model_hparams,
        "exploration_hparams": exploration_hparams,
        "logging_frequency": logging_frequency,
        "num_experiments": num_experiments,
    }
    # return [
    #     dataset, training_mode, nn_params, linear_model_hparams,
    #     exploration_hparams, logging_frequency, num_experiments
    # ]


working_directory = "/checkpoint/apacchiano/"
partition = "learnfair"
gpus_per_node = 1
# ntasks_per_node = 1
ntasks_per_node = 5
nodes = 1


cluster_config = {
    "num_gpus": 2,
    "array_parallelism": 20,
}

args = get_parallel_args() if PARALLEL else get_args()
copy_and_run_with_config(
    run_and_plot,
    args,
    working_directory,
    parallel=PARALLEL,
    array_parallelism=20,
    job_name=JOB_NAME,
    time="72:00:00",
    partition=partition,
    gpus_per_node=gpus_per_node,
    ntasks_per_node=ntasks_per_node,
    cpus_per_task=1,
    # mem="470GB",
    mem="40GB",
    nodes=nodes,
    # constraint="volta32gb",
)
