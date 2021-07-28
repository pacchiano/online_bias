import numpy as np
import matplotlib.pyplot as plt
import torch

from dataclasses import dataclass


@dataclass
class Breakdown:
    fpr: float
    fnr: float
    weight_norm: float


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, MLP=True, fit_intercept = False):
        super(Feedforward, self).__init__()
        self.MLP = MLP
        self.input_size = input_size
        self.sigmoid = torch.nn.Sigmoid()

        self.fit_intercept = fit_intercept
        if self.fit_intercept and self.MLP:
            raise ValueError("Both fit intercept and MLP are on")


        if self.MLP:
            self.hidden_size = hidden_size
            # TODO?
            if torch.cuda.is_available():
                self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size).to('cuda')
                self.relu = torch.nn.ReLU().to('cuda')
                self.fc2 = torch.nn.Linear(self.hidden_size, 1).to('cuda')
            else:
                self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(self.hidden_size, 1)

        else:
            if torch.cuda.is_available():
                self.fc1 = torch.nn.Linear(self.input_size, 1, bias=False).to('cuda')
            else:
                self.fc1 = torch.nn.Linear(self.input_size, 1, bias=False)

    def forward(self, x, inverse_data_covariance=[], alpha=0):
        # TODO
        if torch.cuda.is_available():
            x = x.to('cuda')
    
        if self.MLP:
            hidden = self.fc1(x)
            representation = self.relu(hidden)
            output = self.fc2(representation)

            # return output, relu
        else:
            representation = x
            output = self.fc1(x)

        if len(inverse_data_covariance) != 0:
            # IPython.embed()
            # raise ValueError("asdlfkm")
            if torch.cuda.is_available():
                inverse_data_covariance = inverse_data_covariance.float().to('cuda')



            output = torch.squeeze(output) + alpha * torch.sqrt(
                torch.matmul(
                    representation,
                    torch.matmul(inverse_data_covariance.float(), representation.t()),
                ).diag()
            )

        output = self.sigmoid(output)

        return output, representation


#     return self.__sigmoid(torch.mv(batch_X.float(), self.theta) + torch.from_numpy(self.alpha*self.__inverse_covariance_norm(batch_X, inverse_data_covariance)))#.numpy()


# def representation(self, x):
#     hidden = self.fc1(x)
#     relu = self.relu(hidden)
#     #output = self.fc2(relu)
#     return relu


class TorchBinaryLogisticRegression:
    def __init__(
        self,
        random_init=False,
        dim=None,
        alpha=1,
        MLP=True,
        representation_layer_size=100,
    ):
        self.theta = None
        self.random_init = random_init
        self.alpha = alpha
        self.MLP = MLP
        self.representation_layer_size = representation_layer_size
        self.criterion = torch.nn.BCELoss()


        if dim is not None:
            self.network = Feedforward(dim, representation_layer_size, MLP)
            self.dim = dim
        else:
            raise ValueError("Dimension was none when __init__ binary logistic regression model")


    def reinitialize_model(self):
        self.network = Feedforward(self.dim, self.representation_layer_size, self.MLP)

    def initialize_gaussian(self):
        with torch.no_grad():
            for parameter in self.network.parameters():
                parameter.copy_(torch.normal(0, 0.1, parameter.shape))


    def initialize_model(self, data_dim):
        # if dim == None:
        # if self.MLP:
        # if self.fit_intercept:
        #       batch_X = self.__add_intercept(batch_X)
        self.network = Feedforward(
            data_dim , self.representation_layer_size, self.MLP
        )

    def __inverse_covariance_norm(self, batch_X, inverse_covariance):
        square_norm = np.dot(np.dot(batch_X, inverse_covariance), np.transpose(batch_X))
        return np.diag(np.sqrt(square_norm))


    def get_representation(self, batch_X):
        _, representations = self.network(batch_X.float())
        return representations

    def get_loss(self, batch_X, batch_y):
        if torch.cuda.is_available():
            self.network.to('cuda')
        prob_predictions, _ = self.network(batch_X.float())  # .squeeze()

        return self.criterion(
            torch.squeeze(prob_predictions), torch.squeeze(batch_y.float())
        )

    def predict_prob(self, batch_X, inverse_data_covariance=[]):
        prob_predictions, _ = self.network(
            batch_X.float(),
            inverse_data_covariance=inverse_data_covariance,
            alpha=self.alpha,
        )  # .squeeze()
        return torch.squeeze(prob_predictions)

    def get_predictions(self, batch_X, threshold, inverse_data_covariance=[]):
        prob_predictions = self.predict_prob(batch_X, inverse_data_covariance)
        thresholded_predictions = prob_predictions > threshold
        return thresholded_predictions


    def get_thresholded_predictions(self, batch_X, threshold, inverse_data_covariance=[]):
        prob_predictions = self.predict_prob(batch_X, inverse_data_covariance)
        thresholded_predictions = prob_predictions > threshold
        return thresholded_predictions


    def get_accuracy(self, batch_X, batch_y, threshold, inverse_data_covariance=[]):
        thresholded_predictions = self.get_predictions(
            batch_X, threshold, inverse_data_covariance
        )
        boolean_predictions = thresholded_predictions == batch_y
        return (boolean_predictions * 1.0).mean()

    def get_breakdown(self, batch_X, batch_y, threshold, inverse_data_covariance=[]):
        thresholded_predictions = self.get_predictions(
            batch_X, threshold, inverse_data_covariance
        )
        # False Positive
        false_positive = torch.bitwise_and(thresholded_predictions == 1, batch_y == 0).float()
        false_negative = torch.bitwise_and(thresholded_predictions == 0, batch_y == 1).float()
        with torch.no_grad():
            norm = (torch.linalg.norm(self.network.fc1.weight) + torch.linalg.norm(self.network.fc2.weight)).item()
        return Breakdown(false_positive.mean(), false_negative.mean(), norm)

    def get_special_breakdown(self, batch_X, batch_y, threshold, inverse_data_covariance=[]):
        true_pos = torch.nonzero(batch_y == 1).squeeze(dim=1)
        true_negs = torch.nonzero(batch_y == 0).squeeze(dim=1)
        # TODO
        # pos_accept = torch.tensor(10.0)
        # neg_accept = torch.tensor(-10.0)
        pos_predictions = []
        neg_predictions = []
        if len(true_pos) > 0:
            pos_predictions = self.get_predictions(
                batch_X[true_pos], threshold, inverse_data_covariance
            ).float()
            # pos_accept = pos_predictions.mean()
        if len(true_negs) > 0:
            neg_predictions = self.get_predictions(
                batch_X[true_negs], threshold, inverse_data_covariance
            ).float()
            # neg_accept = neg_predictions.mean()
        # return pos_accept, neg_accept
        n_accepts_pos = 0 if len(true_pos) == 0 else torch.nonzero(pos_predictions == 1).size()[0]
        n_accepts_neg = 0 if len(true_negs) == 0 else torch.nonzero(neg_predictions == 1).size()[0]
        return [
            (n_accepts_pos, len(true_pos)),
            (n_accepts_neg, len(true_negs)),
        ]

    def plot(self, x_min, x_max, num_points=100):
        raise ValueError("plotting not supported")
        x_space = np.linspace(x_min, x_max, num_points)
        y_values = []
        if self.theta.shape[0] == 2:
            for x in x_space:
                y_values.append(-self.theta[0] / self.theta[1] * x)
        elif self.theta.shape[0] == 3:
            for x in x_space:
                y_values.append((-self.theta[0] * x - self.theta[2]) / self.theta[1])
        else:
            raise ValueError("Plotting not supported")
            return 0
        y_values = np.array(y_values)
        plt.plot(x_space, y_values, color="black", label="classifier")


def get_predictions(global_batch, protected_batches, model, inverse_data_covariance=[]):
    batch_X, batch_y = global_batch
    protected_predictions = [
        model.predict_prob(protected_batch[0], inverse_data_covariance)
        for protected_batch in protected_batches
    ]
    global_prediction = model.predict_prob(batch_X, inverse_data_covariance)
    return global_prediction, protected_predictions


def get_error_breakdown(global_batch, model, threshold):
    batch_X, batch_y = global_batch
    breakdown = model.get_breakdown(batch_X, batch_y, threshold)
    return breakdown


# def get_breakdown_no_model(self, batch, preds):
def get_breakdown_no_model(batch):
    # False Positive
    batch_X, batch_y = batch
    preds = torch.ones_like(batch_y)
    false_positive = torch.bitwise_and(preds == 1, batch_y == 0).float()
    false_negative = torch.bitwise_and(preds == 0, batch_y == 1).float()
    return Breakdown(false_positive.mean(), false_negative.mean(), 0)


def get_special_breakdown(batch, model, threshold):
    batch_X, batch_y = batch
    breakdown = model.get_special_breakdown(batch_X, batch_y, threshold)
    return breakdown


def get_accuracies(global_batch, protected_batches, model, threshold):
    batch_X, batch_y = global_batch

    accuracies_list = model.get_accuracy(batch_X, batch_y, threshold)

    protected_accuracies_list = [
        model.get_accuracy(protected_batch[0], protected_batch[1], threshold)
        for protected_batch in protected_batches
    ]
    return accuracies_list, protected_accuracies_list


def get_accuracies_simple(global_batch, model, threshold):
    batch_X, batch_y = global_batch

    accuracies_list = model.get_accuracy(batch_X, batch_y, threshold)

    return accuracies_list
