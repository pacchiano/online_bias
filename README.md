# Neural Pseudo-Label Optimism for the Bank Loan Problem

This repository is the official implementation of Neural Pseudo-Label Optimism for the Bank Loan Problem.

## Requirements

NOTE: To run our experiments, we require pytorch to be compiled with GPU support, and for your machine to have at least one GPU.
To effectively and efficiently reproduce our multi-experiment runs, we recommend a system with 5 (or N_EXPERIMENTS) GPUs.
This is much simpler on a Linux system, which supports pytorch CUDA binaries.
Instructions are provided for Linux.

To install requirements:

```setup
conda env create -f environment.yml
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

## Experiments

To run the experiments in the paper, run this command:

```experiments
python pytorch_experiments.py True True 5
```

This will run 5 experiments in parallel. To run only one replicate, simply run:
```experiment
python pytorch_experiments.py True True 1
```

To disable the parallelization of Ray:
```experiment
python pytorch_experiments.py True False 1
```

Hyperparameters are already specified in the model file, and can easily be viewed inside of `pytorch_experiments.py`.

## Results

Our algorithm achieves the following performances:

| Dataset          | Cumulative Regret@T=2000 | Std. Dev of Cumulative Regret@T=2000 |
| ---------------- |------------------------- | -----------------------------------  |
| Adult            |            6.145         |      1.529                           |
| Bank             |            0.736         |      0.528                           |
| MNIST            |            1.711         |      0.406                           |

These results are output to the `experiment_results` folder after running the experiment code.
To get these printed, please copy `final_results/analyze.py` to the dataset folder you wish to analyze.
E.g. for Adult: `experiment_results/Adult/PLOT/data/`. From that directory, run `python analyze.py`, and the cumulative regret (mean+stddev) of the method will be printed.
