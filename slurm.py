#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import pathlib
import shutil

import submitit

from pytorch_experiments import main


def init_and_run(run_fn, run_config):
    os.environ["RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["NODE_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    run_fn(run_config)


def copy_and_run_with_config(run_fn, run_config, directory, **cluster_config):
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

run_config = {

}

copy_and_run_with_config(
    main,
    args,
    args.working_directory,
    job_name=args.job_name,
    time="72:00:00",
    partition=args.partition,
    gpus_per_node=args.gpus,
    ntasks_per_node=args.gpus,
    cpus_per_task=10,
    # mem="470GB",
    mem="40GB",
    nodes=args.num_nodes,
    # constraint="volta32gb",
)
