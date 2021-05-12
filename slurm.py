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


def init_and_run(run_fn, run_config):
    os.environ["RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["NODE_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    run_fn(**run_config)


def copy_and_run_with_config(run_fn, run_config, directory, **cluster_config):
    print("Let's use slurm!")
    working_directory = pathlib.Path(directory) / cluster_config["job_name"]
    ignore_list = [
        "lightning_logs",
        "logs",
        "checkpoints",
        "experiments",
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
    job = executor.submit(init_and_run, run_fn, run_config)
    print(f"job_id: {job}")


cluster_config = {
    "num_gpus": 2,
    "array_parallelism": 20,
}


def get_args():
    dataset = "MultiSVM"
    training_mode = "full_minimization"
    nn_params = NNParams()
    linear_model_hparams = LinearModelHparams()
    exploration_hparams = ExplorationHparams()
    num_experiments = 5
    logging_frequency = 1000
    return {
        "dataset": dataset,
        "training_mode": training_mode,
        "nn_params": nn_params,
        "linear_model_hparams": linear_model_hparams,
        "exploration_hparams": exploration_hparams,
        "logging_frequency": logging_frequency,
        "num_experiments": num_experiments,
    }


args = get_args()
working_directory = "/checkpoint/apacchiano/"
job_name = "fair_bandits_test"
# partition = "scavenge"
partition = "learnfair"
gpus_per_node = 1
ntasks_per_node = 1
nodes = 1


copy_and_run_with_config(
    run_and_plot,
    args,
    working_directory,
    job_name=job_name,
    time="72:00:00",
    partition=partition,
    gpus_per_node=gpus_per_node,
    ntasks_per_node=ntasks_per_node,
    cpus_per_task=10,
    # mem="470GB",
    mem="40GB",
    nodes=nodes,
    # constraint="volta32gb",
)
