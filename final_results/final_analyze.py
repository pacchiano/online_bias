import os
import pandas as pd
import pickle
import sys

sys.path.append(
    "/private/home/apacchiano/OnlineBias"
)

from pytorch_experiments import ExperimentResults


datasets = ["Adult", "Bank", "MNIST", "MultiSVM"]
results = pd.DataFrame(columns=["Exp", "Dataset", "Regret", "Regret_STD"])
for exp in os.listdir():
    for dataset in datasets:
        path = f"/checkpoint/apacchiano/final_results/bank/{exp}/experiment_results/{dataset}/data"
        try:
            filename = os.path.join(path, "data_dump.p")
            print(filename)
            with open(filename, 'rb') as f:
                x = pickle.load(f)
                print([round(a, 3) for a in list(x[2].mean_train_cum_regret_averages)])
                print([round(a, 3) for a in list(x[2].std_train_cum_regret_averages)])
                regret = [round(a, 3) for a in list(x[2].mean_train_cum_regret_averages)]
                regret_std = [round(a, 3) for a in list(x[2].std_train_cum_regret_averages)]
                results = results.append(
                    {
                        "Exp": exp, "Dataset": dataset, "Regret": regret, "Regret_STD": regret_std
                    }, ignore_index=True
                )
        except Exception as e:
            print(e)
            pass
results.to_csv('results.csv')
