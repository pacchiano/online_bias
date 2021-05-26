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
    NUM_EXPERIMENTS,
)

PARALLEL = True
# FAST = True
# VERSION = "fast_ray_distr"
FAST = False
T = 300
BATCH = 32
EPS_GREEDY = True
GREEDY = False
METHOD = "pseudolabel_"
if EPS_GREEDY:
    METHOD = "eps_greedy_"
if GREEDY:
    METHOD = "greedy_"
# DECAY = 0.001
DECAY = 0.0
VERSION = f"_{T}t_{METHOD}ray_no_warm_batch_{BATCH}_decay_{DECAY}_gpu"
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
        # [30, 100, 1000]
        if FAST:
            nn_param.max_num_steps = 5
            nn_param.baseline_steps = 1000
            nn_param.batch_size = 1
        else:
            # TODO
            nn_param.max_num_steps = T
            nn_param.batch_size = BATCH
            nn_param.weight_decay = DECAY
    linear_model_hparams = [LinearModelHparams()] * len(datasets)
    exploration_hparam = ExplorationHparams()
    if EPS_GREEDY:
        exploration_hparam.decision_type = "simple"
        exploration_hparam.epsilon_greedy = True
    elif GREEDY:
        exploration_hparam.decision_type = "simple"
        exploration_hparam.epsilon_greedy = False
    exploration_hparams = [exploration_hparam] * len(datasets)
    num_experiments = [NUM_EXPERIMENTS] * len(datasets)
    logging_frequency = [int(T / 5)] * len(datasets)
    return [
        datasets, training_modes, nn_param_list, linear_model_hparams,
        exploration_hparams, num_experiments, logging_frequency
    ]


working_directory = "/checkpoint/apacchiano/"
partition = "prioritylab"
gpus_per_node = 5
ntasks_per_node = 1
# ntasks_per_node = 5
# TODO: not needed, this is only useful for distr.
# nodes = 3
# Job array can easily handle this.
nodes = 1


args = get_parallel_args()
copy_and_run_with_config(
    run_and_plot,
    args,
    working_directory,
    parallel=PARALLEL,
    array_parallelism=20,
    job_name=JOB_NAME,
    time="72:00:00",
    comment="Neurips Deadline",
    partition=partition,
    gpus_per_node=gpus_per_node,
    ntasks_per_node=ntasks_per_node,
    gpus_per_task=5,
    cpus_per_task=5,
    # mem="470GB",
    mem="100GB",
    nodes=nodes,
    # constraint="volta32gb",
)
