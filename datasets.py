import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
from torchvision import datasets, transforms
from datasets_fairness import (
    AdultParams,
    BankParams,
    CrimeParams,
    GermanParams,
    collect_adult_protected_xy,
    collect_bank_protected_xy,
    collect_crime_protected_xy,
    collect_german_protected_xy,
    read_and_preprocess_adult_data_uai,
    read_and_preprocess_german_data,
    read_and_preprocess_bank_data,
    read_and_preprocess_crime_data,
)


# @title Data utilities
class DataSet:
    def __init__(self, dataset, labels, num_classes=2):
        self.num_datapoints = dataset.shape[0]
        self.dimension = dataset.shape[1]
        self.random_state = 0
        self.dataset = dataset
        self.labels = labels
        self.num_classes = num_classes

    def get_batch(self, batch_size):
        if batch_size > self.num_datapoints:
            X = self.dataset.values
            Y = self.labels.values
        else:
            X = self.dataset.sample(batch_size, random_state=self.random_state).values
            Y = self.labels.sample(batch_size, random_state=self.random_state).values
        # Y_one_hot = np.zeros((Y.shape[0], self.num_classes))
        # for i in range(self.num_classes):
        #   Y_one_hot[:, i] = (Y == i)*1.0
        self.random_state += 1

        return (torch.from_numpy(X).to('cuda'), torch.from_numpy(Y).to('cuda'))


class GrowingNumpyDataSet:
    def __init__(self):
        self.dataset_X = None
        self.dataset_Y = None
        self.last_data_addition = None
        self.random_state = 0

    def get_size(self):
        if self.dataset_Y is None:
            return 0
        return len(self.dataset_Y)

    def add_data(self, X, Y):
        if self.dataset_X is None and self.dataset_Y is None:
            self.dataset_X = X
            self.dataset_Y = Y
        else:
            self.dataset_X = torch.cat((self.dataset_X, X), dim=0)
            self.dataset_Y = torch.cat((self.dataset_Y, Y), dim=0)
        # print("shapes")
        # print(self.dataset_X.shape)
        # print(X.shape)
        # print("datasets")
        # print(self.dataset_X)
        # print(X)
        # print(self.dataset_X.shape)

        self.last_data_addition = X.shape[0]

    def pop_last_data(self):
        if self.dataset_X.shape[0] == self.last_data_addition:
            self.dataset_X = None
            self.dataset_Y = None

        else:
            # self.dataset_X = self.dataset_X[: -self.last_data_addition, :]
            # self.dataset_Y = self.dataset_Y[: -self.last_data_addition, :]
            self.dataset_X = self.dataset_X[: -self.last_data_addition]
            self.dataset_Y = self.dataset_Y[: -self.last_data_addition]

    def get_batch(self, batch_size):
        if self.dataset_X is None:
            X = torch.empty(0)
            Y = torch.empty(0)
        elif batch_size > self.dataset_X.shape[0]:
            X = self.dataset_X
            Y = self.dataset_Y
        else:
            indices = random.sample(range(self.dataset_X.shape[0]), batch_size)
            indices = torch.tensor(indices)
            X = self.dataset_X[indices]
            Y = self.dataset_Y[indices]
        self.random_state += 1
        return (X, Y)


class MNISTDataset:
    def __init__(self, train, batch_size, symbol):

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.symbol = symbol
        self.batch_size = batch_size
        self.dataset = datasets.MNIST(
            "./", train=train, download=False, transform=transform
        )
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True
        )

    def get_batch(self, batch_size):
        if batch_size != self.batch_size:
            raise ValueError(
                "Provided batch size does not agree with the stored batch size. MNIST."
            )
        [X, Y] = next(iter(self.data_loader))
        Y = (Y == self.symbol) * 1.0
        X = X.view(self.batch_size, -1)
        if torch.cuda.is_available():
            # print("Getting gpu")
            X = X.to('cuda')
            Y = Y.to('cuda')
        return (X, Y)
        # return (X.numpy(), Y.numpy())


class MixtureGaussianDataset:
    def __init__(
        self,
        means,
        variances,
        probabilities,
        theta_stars,
        num_classes=2,
        max_batch_size=10000,
        kernel=lambda a, b: np.dot(a, b),
    ):
        self.means = means
        self.variances = variances
        self.probabilities = probabilities
        self.num_classes = num_classes
        self.theta_stars = theta_stars
        self.cummulative_probabilities = np.zeros(len(probabilities))
        cum_prob = 0
        for i, prob in enumerate(self.probabilities):
            cum_prob += prob
            self.cummulative_probabilities[i] = cum_prob
        self.dimension = theta_stars[0].shape[0]
        self.max_batch_size = max_batch_size
        self.kernel = kernel

    def get_batch(self, batch_size):
        batch_size = min(batch_size, self.max_batch_size)
        X = []
        Y = []
        for _ in range(batch_size):
            val = np.random.random()
            index = 0
            while index <= len(self.cummulative_probabilities) - 1:
                if val < self.cummulative_probabilities[index]:
                    break
                index += 1

            x = np.random.multivariate_normal(
                self.means[index], np.eye(self.dimension) * self.variances[index]
            )
            logit = self.kernel(x, self.theta_stars[index])
            y_val = 1 / (1 + np.exp(-logit))
            y = (np.random.random() >= y_val) * 1.0
            X.append(x)
            Y.append(y)
        X = np.array(X)
        Y = np.array(Y)
        return (X, Y)


class SVMDataset:
    def __init__(
        self,
        means,
        variances,
        probabilities,
        class_list_per_center,
        num_classes=2,
        max_batch_size=10000,
    ):
        self.means = means
        self.variances = variances
        self.probabilities = probabilities
        self.num_classes = num_classes
        self.class_list_per_center = class_list_per_center
        self.cummulative_probabilities = np.zeros(len(probabilities))
        cum_prob = 0
        for i, prob in enumerate(self.probabilities):
            cum_prob += prob
            self.cummulative_probabilities[i] = cum_prob
        self.max_batch_size = max_batch_size
        self.num_groups = len(self.means)
        self.dim = self.means[0].shape[0]

    def get_batch(self, batch_size, verbose=False):
        batch_size = min(batch_size, self.max_batch_size)
        X = []
        Y = []
        indices = []
        for _ in range(batch_size):
            val = np.random.random()
            index = 0
            while index <= len(self.cummulative_probabilities) - 1:
                if val < self.cummulative_probabilities[index]:
                    break
                index += 1

            x = np.random.multivariate_normal(
                self.means[index], np.eye(self.dim) * self.variances[index]
            )
            y = self.class_list_per_center[index]
            X.append(x)
            Y.append(y)
            indices.append(index)
        X = np.array(X)
        Y = np.array(Y)
        indices = np.array(indices)

        if torch.cuda.is_available():
            X = torch.from_numpy(X).to('cuda')
            Y = torch.from_numpy(Y).to('cuda')

        if verbose:
            return (X, Y, indices)
        else:
            return (X, Y)

    def plot(self, batch_size, model=None, names=[]):
        if names == []:
            names = ["" for _ in range(self.num_groups)]
        if self.dim != 2:
            print("Unable to plot the dataset")
        else:
            colors = [
                "blue",
                "red",
                "green",
                "yellow",
                "black",
                "orange",
                "purple",
                "violet",
                "gray",
            ]
            (X, Y, indices) = self.get_batch(batch_size, verbose=True)
            # print("xvals ", X, "yvals ", Y)
            min_x = float("inf")
            max_x = -float("inf")
            for i in range(self.num_groups):
                X_filtered_0 = []
                X_filtered_1 = []
                for j in range(len(X)):
                    if indices[j] == i:
                        X_filtered_0.append(X[j][0])
                        if X[j][0] < min_x:
                            min_x = X[j][0]
                        if X[j][0] > max_x:
                            max_x = X[j][0]
                        X_filtered_1.append(X[j][1])

                plt.plot(
                    X_filtered_0,
                    X_filtered_1,
                    "o",
                    color=colors[i],
                    label="{} {}".format(self.class_list_per_center[i], names[i]),
                )
            if model is not None:
                # Plot line
                model.plot(min_x, max_x)
            plt.grid(True)
            plt.legend(loc="lower right")
            # IPython.embed()

            # plt.show()


def get_batches(protected_datasets, global_dataset, batch_size):
    global_batch = global_dataset.get_batch(batch_size)

    protected_batches = [
        protected_dataset.get_batch(batch_size)
        for protected_dataset in protected_datasets
    ]
    return global_batch, protected_batches


def get_dataset(dataset, batch_size, test_batch_size):

    if dataset == "Mixture":
        PROTECTED_GROUPS = ["A", "B", "C", "D"]
        d = 20
        means = [
            -10 * np.arange(d) / np.linalg.norm(np.ones(d)),
            np.zeros(d),
            10 * np.arange(d) / np.linalg.norm(np.arange(d)),
            np.ones(d) / np.linalg.norm(np.ones(d)),
        ]
        variances = [0.4, 0.41, 0.41, 0.41]
        theta_stars = [np.zeros(d), np.zeros(d), np.zeros(d), np.zeros(d)]
        probabilities = [0.3, 0.1, 0.5, 0.1]
        kernel = lambda a, b: 0.1 * np.dot(a - b, a - b) - 1

        protected_datasets_train = [
            MixtureGaussianDataset(
                [means[i]], [variances[i]], [1], [theta_stars[i]], kernel=kernel
            )
            for i in range(len(PROTECTED_GROUPS))
        ]
        protected_datasets_test = [
            MixtureGaussianDataset(
                [means[i]], [variances[i]], [1], [theta_stars[i]], kernel=kernel
            )
            for i in range(len(PROTECTED_GROUPS))
        ]

        train_dataset = MixtureGaussianDataset(
            means, variances, probabilities, theta_stars, kernel=kernel
        )
        test_dataset = MixtureGaussianDataset(
            means, variances, probabilities, theta_stars, kernel=kernel
        )
    elif dataset == "Adult":
        # PROTECTED_GROUPS = [
        #     'Female_White', 'Female_Black', 'Male_White', 'Male_Black'
        # ]''

        joint_protected_groups = False
        if joint_protected_groups:
            PROTECTED_GROUPS = AdultParams.JOINT_PROTECTED_GROUPS
        else:
            PROTECTED_GROUPS = AdultParams.PROTECTED_GROUPS

        (
            dataframe_all_train,
            dataframe_all_test,
            feature_names,
        ) = read_and_preprocess_adult_data_uai(remove_missing=False)
        # dataframe_all_train = dataframe_all_train.sample(data_size, random_state=random_state)

        # Identify portions of the data corresponding to particuar values of specific protected attributes.
        # REMOVED

        # Split all data into features and labels.
        x_all_train = dataframe_all_train[feature_names]
        y_all_train = dataframe_all_train[AdultParams.LABEL_COLUMN]
        x_all_test = dataframe_all_test[feature_names]
        y_all_test = dataframe_all_test[AdultParams.LABEL_COLUMN]

        train_dataset = DataSet(x_all_train, y_all_train)
        test_dataset = DataSet(x_all_test, y_all_test)

        xy_protected_train = collect_adult_protected_xy(
            x_all_train, y_all_train, PROTECTED_GROUPS
        )
        xy_protected_test = collect_adult_protected_xy(
            x_all_test, y_all_test, PROTECTED_GROUPS
        )

        protected_datasets_train = [
            DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_train
        ]
        protected_datasets_test = [
            DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_test
        ]

        # protected_datasets_train = [DataSet(x_vals, y_vals) for (x_vals, y_vals) in protected_dataframes_train]
        # protected_datasets_test = [DataSet(x_vals, y_vals) for (x_vals, y_vals) in protected_dataframes_test]

        # train_dataset = DataSet(X_train_adult_df, y_train_adult_df)
        # test_dataset = DataSet(X_test_adult_df, y_test_adult_df)

    elif dataset == "German":
        PROTECTED_GROUPS = GermanParams.PROTECTED_THRESHOLDS

        # LOAD ALL THE DATA
        (
            dataframe_all_train,
            dataframe_all_test,
            feature_names,
        ) = read_and_preprocess_german_data()
        # dataframe_all_train = dataframe_all_train.sample(data_size, random_state=random_state)

        # Identify portions of the data corresponding to particuar values of specific
        # protected attributes.
        # REMOVED

        # Split all data into features and labels.
        x_all_train = dataframe_all_train[feature_names]
        y_all_train = dataframe_all_train[GermanParams.LABEL_COLUMN]
        x_all_test = dataframe_all_test[feature_names]
        y_all_test = dataframe_all_test[GermanParams.LABEL_COLUMN]

        # In utilities_final.py: Dataset class to be able to sample batches
        train_dataset = DataSet(x_all_train, y_all_train)
        test_dataset = DataSet(x_all_test, y_all_test)

        xy_protected_train = collect_german_protected_xy(
            x_all_train, y_all_train, PROTECTED_GROUPS
        )
        xy_protected_test = collect_german_protected_xy(
            x_all_test, y_all_test, PROTECTED_GROUPS
        )

        protected_datasets_train = [
            DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_train
        ]
        protected_datasets_test = [
            DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_test
        ]

    elif dataset == "Bank":

        PROTECTED_GROUPS = BankParams.PROTECTED_GROUPS

        # LOAD ALL THE DATA
        (
            dataframe_all_train,
            dataframe_all_test,
            feature_names,
        ) = read_and_preprocess_bank_data()
        # dataframe_all_train = dataframe_all_train.sample(data_size, random_state=random_state)

        # Identify portions of the data corresponding to particuar values of specific
        # protected attributes.
        # REMOVED

        # Split all data into features and regression targets.
        x_all_train = dataframe_all_train[feature_names]
        y_all_train = dataframe_all_train[BankParams.LABEL_COLUMN]
        x_all_test = dataframe_all_test[feature_names]
        y_all_test = dataframe_all_test[BankParams.LABEL_COLUMN]

        # In utilities_final.py: Dataset class to be able to sample batches
        train_dataset = DataSet(x_all_train, y_all_train)
        test_dataset = DataSet(x_all_test, y_all_test)

        xy_protected_train = collect_bank_protected_xy(
            x_all_train, y_all_train, PROTECTED_GROUPS
        )
        xy_protected_test = collect_bank_protected_xy(
            x_all_test, y_all_test, PROTECTED_GROUPS
        )

        protected_datasets_train = [
            DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_train
        ]
        protected_datasets_test = [
            DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_test
        ]

    elif dataset == "Crime":

        PROTECTED_GROUPS = CrimeParams.PROTECTED_GROUPS

        # LOAD ALL THE DATA
        (
            dataframe_all_train,
            dataframe_all_test,
            feature_names,
        ) = read_and_preprocess_crime_data()
        # dataframe_all_train = dataframe_all_train.sample(data_size, random_state=random_state)

        # Split all data into features and labels.
        x_all_train = dataframe_all_train[feature_names]
        y_all_train = dataframe_all_train[CrimeParams.LABEL_COLUMN]
        x_all_test = dataframe_all_test[feature_names]
        y_all_test = dataframe_all_test[CrimeParams.LABEL_COLUMN]

        # In utilities_final.py: Dataset class to be able to sample batches
        train_dataset = DataSet(x_all_train, y_all_train)
        test_dataset = DataSet(x_all_test, y_all_test)

        xy_protected_train = collect_crime_protected_xy(
            x_all_train, y_all_train, PROTECTED_GROUPS
        )
        xy_protected_test = collect_crime_protected_xy(
            x_all_test, y_all_test, PROTECTED_GROUPS
        )

        protected_datasets_train = [
            DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_train
        ]
        protected_datasets_test = [
            DataSet(x_vals, y_vals) for (x_vals, y_vals) in xy_protected_test
        ]

    elif dataset == "MNIST":
        PROTECTED_GROUPS = ["None"]
        protected_datasets_train = [
            MNISTDataset(train=True, batch_size=batch_size, symbol=5)
        ]
        train_dataset = MNISTDataset(train=True, batch_size=batch_size, symbol=5)

        protected_datasets_test = [
            MNISTDataset(train=False, batch_size=test_batch_size, symbol=5)
        ]
        test_dataset = MNISTDataset(train=False, batch_size=test_batch_size, symbol=5)

    elif dataset == "MultiSVM":
        PROTECTED_GROUPS = ["A", "B", "C", "D"]
        d = 2
        means = [
            np.array([0, 5]),
            np.array([0, 0]),
            np.array([5, -2]),
            np.array([5, 5]),
        ]
        variances = [0.5, 0.5, 0.5, 0.5]
        probabilities = [0.3, 0.3, 0.2, 0.2]
        class_list_per_center = [1, 0, 1, 0]

        protected_datasets_train = [
            SVMDataset([means[i]], [variances[i]], [1], [class_list_per_center[i]])
            for i in range(len(PROTECTED_GROUPS))
        ]
        protected_datasets_test = [
            SVMDataset([means[i]], [variances[i]], [1], [class_list_per_center[i]])
            for i in range(len(PROTECTED_GROUPS))
        ]

        train_dataset = SVMDataset(
            means, variances, probabilities, class_list_per_center
        )
        test_dataset = SVMDataset(
            means, variances, probabilities, class_list_per_center
        )

    elif dataset == "SVM":
        PROTECTED_GROUPS = ["A", "B"]
        d = 2
        means = [
            -np.arange(d) / np.linalg.norm(np.arange(d)),
            np.ones(d) / np.linalg.norm(np.ones(d)),
        ]
        variances = [1, 0.1]
        probabilities = [0.5, 0.5]
        class_list_per_center = [0, 1]

        protected_datasets_train = [
            SVMDataset([means[i]], [variances[i]], [1], [class_list_per_center[i]])
            for i in range(len(PROTECTED_GROUPS))
        ]
        protected_datasets_test = [
            SVMDataset([means[i]], [variances[i]], [1], [class_list_per_center[i]])
            for i in range(len(PROTECTED_GROUPS))
        ]

        train_dataset = SVMDataset(
            means, variances, probabilities, class_list_per_center
        )
        test_dataset = SVMDataset(
            means, variances, probabilities, class_list_per_center
        )
    else:
        raise ValueError("Unrecognized dataset")

    return (
        protected_datasets_train,
        protected_datasets_test,
        train_dataset,
        test_dataset,
    )
