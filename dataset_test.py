from datasets import SVMDataset, get_dataset
import matplotlib.pyplot as plt
import IPython




# protected_datasets_train, protected_datasets_test, train_dataset, test_dataset = get_dataset("MultiSVM")

# batch_size = 100
# train_dataset.plot( batch_size, model = None, names = ["A", "B", "C", "D"])

# plt.savefig("./experiment_results/dataset_test.png")






protected_datasets_train, protected_datasets_test, train_dataset, test_dataset = get_dataset("Adult", 30, 10)