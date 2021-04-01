from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib 
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats

import IPython
import requests
import pandas as pd
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
import numpy.random as npr
from scipy.stats import wasserstein_distance, ks_2samp
# from sklearn.linear_model import LogisticRegression

#%matplotlib inline


class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size, MLP = True):
            super(Feedforward, self).__init__()
            self.MLP = MLP
            self.input_size = input_size
            self.sigmoid = torch.nn.Sigmoid()

            if self.MLP:

              #self.input_size = input_size
              self.hidden_size  = hidden_size
              self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
              self.relu = torch.nn.ReLU()
              self.fc2 = torch.nn.Linear(self.hidden_size, 1)
              #self.sigmoid = torch.nn.Sigmoid()

              

            else:
              self.fc1 = torch.nn.Linear(self.input_size, 1, bias = False)



        def forward(self, x, inverse_data_covariance = [], alpha = 0):
            if self.MLP:
              hidden = self.fc1(x)
              representation = self.relu(hidden)
              output = self.fc2(representation)

              #return output, relu
            else:
              representation= x
              output = self.fc1(x)

            if len(inverse_data_covariance) != 0:
                # IPython.embed()
                # raise ValueError("asdlfkm")
                output = torch.squeeze(output) + alpha*torch.sqrt(torch.matmul( representation, torch.matmul(inverse_data_covariance.float(), representation.t() )   ).diag() )




            output = self.sigmoid(output)

            return output, representation


      #     return self.__sigmoid(torch.mv(batch_X.float(), self.theta) + torch.from_numpy(self.alpha*self.__inverse_covariance_norm(batch_X, inverse_data_covariance)))#.numpy()


        # def representation(self, x):
        #     hidden = self.fc1(x)
        #     relu = self.relu(hidden)
        #     #output = self.fc2(relu)
        #     return relu

class TorchBinaryLogisticRegression:
    def __init__(self, random_init = False, fit_intercept = True, dim = None, alpha = 1, 
      MLP = True, representation_layer_size = 100):
        self.fit_intercept = fit_intercept
        self.theta = None
        self.random_init = random_init
        self.alpha = alpha        
        self.MLP = MLP
        self.representation_layer_size = representation_layer_size
        self.criterion = torch.nn.BCELoss()

        if dim != None:
          self.network = Feedforward(dim, representation_layer_size, MLP)
          #self.initialize_gaussian()
    
    def initialize_gaussian(self):
      with torch.no_grad():
        for parameter in self.network.parameters():
          parameter.copy_(torch.normal(0,.1,parameter.shape))

    def initialize_model(self, batch_X):
      # if dim == None:
      #if self.MLP:
      if self.fit_intercept:
            batch_X = self.__add_intercept(batch_X)
      self.network = Feedforward(batch_X.shape[1], self.representation_layer_size, self.MLP)
      
      #self.initialize_gaussian()


    def __add_intercept(self, batch_X):

        intercept = np.ones((batch_X.shape[0], 1))        
        return np.concatenate((batch_X, intercept), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))
    def __loss(self, h, y):
        return (-y * torch.log(h) - (1 - y) * torch.log(1 - h)).mean()

    def __inverse_covariance_norm(self, batch_X, inverse_covariance):
      square_norm = np.dot(np.dot(batch_X, inverse_covariance), np.transpose(batch_X))
      return np.diag(np.sqrt(square_norm))

    def __update_batch(self, batch_X):
        if self.fit_intercept:
          return self.__add_intercept(batch_X)
        return batch_X
 
    # def update_batch(self, batch_X):
    #     if self.fit_intercept:
    #       return self.__add_intercept(batch_X)
    #     return batch_X

    def get_representation(self, batch_X):
      batch_X = self.__update_batch(batch_X) 
      batch_X = torch.from_numpy(batch_X)
      _, representations = self.network(batch_X.float())
      return representations


    def get_loss(self, batch_X, batch_y):
      #self.__initialize_theta(batch_X)
      if len(batch_y) == 1:
        IPython.embed()
        raise ValueError("asdflkm")
      batch_X = self.__update_batch(batch_X)    
      batch_X = torch.from_numpy(batch_X)
      batch_y = torch.from_numpy(batch_y)
      # import IPython
      # IPython.embed()
      # raise ValueError("Asdflkm")
      #if self.MLP:
      #prob_predictions, representations =  self.network(batch_X.float())#.squeeze()
      prob_predictions, _ =  self.network(batch_X.float())#.squeeze()

      # import IPython
      # IPython.embed()
      # raise ValueError("Asdflkm")
      #return -torch.mean(batch_y.float()*torch.log(prob_predictions)) - torch.mean((1-batch_y.float())*torch.log(1-prob_predictions))

      return self.criterion(torch.squeeze(prob_predictions), batch_y.float())
      

      # else:
      #   z = torch.mv(batch_X.float(), self.theta)
      #   h = self.__sigmoid(z)
      #   return self.__loss(h, batch_y)
  

    def predict_prob(self, batch_X, inverse_data_covariance = []):
      #self.__initialize_theta(batch_X)
      batch_X = self.__update_batch(batch_X) 
      batch_X = torch.from_numpy(batch_X)

      #if self.MLP:
      #prob_predictions, representations =  self.network(batch_X.float())#.squeeze()
      #if len(inverse_data_covariance) == 0:
      prob_predictions, _ =  self.network(batch_X.float(), inverse_data_covariance = inverse_data_covariance, alpha = self.alpha)#.squeeze()
      #prob_predictions = torch.squeeze(prob_predictions)
      #prob_predictions, _  =  self.network(batch_X.float())#.squeeze()


      return torch.squeeze(prob_predictions)
      # else:

      #   if len(inverse_data_covariance) == 0:     
      #     return self.__sigmoid(torch.mv(batch_X.float(), self.theta))#.numpy()
      #   else:
      #     ### adjust this using the inverse_data_covariance
      #     ### for each datapoint return the max 
      #     return self.__sigmoid(torch.mv(batch_X.float(), self.theta) + torch.from_numpy(self.alpha*self.__inverse_covariance_norm(batch_X, inverse_data_covariance)))#.numpy()


    def get_predictions(self, batch_X, threshold, inverse_data_covariance = []):    
      # if self.MLP:
      #   #self.network.eval()
      #   prob_predictions =  self.network(batch_X.float())
      # else:
      prob_predictions = self.predict_prob(batch_X, inverse_data_covariance )
      
      thresholded_predictions= prob_predictions > threshold
      thresholded_predictions = torch.squeeze(thresholded_predictions)
      thresholded_predictions = thresholded_predictions.numpy()
      return thresholded_predictions

    def get_accuracy( self, batch_X, batch_y, threshold, inverse_data_covariance = []):
      thresholded_predictions = self.get_predictions(batch_X, threshold, inverse_data_covariance)
      boolean_predictions = (thresholded_predictions == batch_y)
      return (boolean_predictions*1.0).mean()


    def plot(self, x_min, x_max, num_points = 100):
      x_space = np.linspace(x_min, x_max, num_points)
      y_values = []
      if self.theta.shape[0] == 2 and not self.fit_intercept:
        for x in x_space:
          y_values.append( -self.theta[0]/self.theta[1]*x )
      elif self.theta.shape[0]==3 and self.fit_intercept:
          for x in x_space:
            y_values.append((-self.theta[0]*x - self.theta[2] )/self.theta[1] )
      else:
        print("Plotting not supported")
        return 0
      y_values = np.array(y_values)
      plt.plot(x_space, y_values, color="black", label = "classifier")





def get_predictions(global_batch, protected_batches, model, inverse_data_covariance = []):
    batch_X, batch_y = global_batch
    #logistic_loss = model.get_loss( batch_X, batch_y)
    #logistic_gradient = model.get_gradient(batch_X, batch_y)

    protected_predictions = [model.predict_prob(protected_batch[0], inverse_data_covariance) for protected_batch in protected_batches]
    global_prediction = model.predict_prob(batch_X, inverse_data_covariance)
    return global_prediction, protected_predictions

def get_accuracies(global_batch, protected_batches, model, threshold):
      batch_X, batch_y = global_batch

      accuracies_list = model.get_accuracy(batch_X, batch_y, threshold)
      protected_accuracies_list= [model.get_accuracy(protected_batch[0], protected_batch[1], threshold) for protected_batch in protected_batches ]
      return accuracies_list, protected_accuracies_list

