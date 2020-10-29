from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib 
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats

import requests
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy.random as npr
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.linear_model import LogisticRegression

#%matplotlib inline


class BinaryLogisticRegression:
    def __init__(self, random_init = False, fit_intercept=True, dim = None):
        self.fit_intercept = fit_intercept
        self.theta = None
        self.random_init = random_init
        if dim != None:
          if self.random_init:
            self.theta = np.random.normal(0, 1, dim + fit_intercept)
          else:
            self.theta = np.zeros(dim + fit_intercept)
          
    
    def __initialize_theta(self, batch_X):
      if self.theta is None:    
        if self.fit_intercept:
          batch_X = self.__add_intercept(batch_X)
        if self.random_init:
          self.theta = np.random.normal(0, 1, batch_X.shape[1])
        else:
          self.theta = np.zeros(batch_X.shape[1])

    def __add_intercept(self, batch_X):
        intercept = np.ones((batch_X.shape[0], 1))
        #print(np.concatenate((batch_X, intercept), axis=1) , " augmented batch")
        #print(np.concatenate((batch_X, intercept), axis=1).shape , " augmented batch shape")
        
        return np.concatenate((batch_X, intercept), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def __inverse_covariance_norm(self, batch_X, inverse_covariance):
      square_norm = np.dot(np.dot(batch_X, inverse_covariance), np.transpose(batch_X))
      return np.diag(np.sqrt(square_norm))

    def __update_batch(self, batch_X):
        if self.fit_intercept:
          return self.__add_intercept(batch_X)
        return batch_X
 
    def update_batch(self, batch_X):
        if self.fit_intercept:
          return self.__add_intercept(batch_X)
        return batch_X


    def get_loss(self, batch_X, batch_y):
      self.__initialize_theta(batch_X)
      batch_X = self.__update_batch(batch_X)    


      z = np.dot(batch_X, self.theta)
      h = self.__sigmoid(z)
      return self.__loss(h, batch_y)
  
    def get_sigmoid_gradients(self, batch_X ):
      self.__initialize_theta( batch_X)
      batch_X = self.__update_batch(batch_X)    

      z = np.dot(batch_X, self.theta)
      h = self.__sigmoid(z)
      gradients = batch_X.T * np.outer(  np.ones( batch_X.shape[1] ),  h)
      return gradients


    def get_gradient(self, batch_X, batch_y , inverse_probabilities = [None] ):
      if inverse_probabilities[0] == None:
        inverse_probabilities = np.ones(batch_X.shape[0])
      self.__initialize_theta( batch_X)
      batch_X = self.__update_batch(batch_X)    

      z = np.dot(batch_X, self.theta)
      h = self.__sigmoid(z)
      gradient = np.dot(batch_X.T, inverse_probabilities*(h-batch_y))/batch_y.size
      return gradient

    def update(self,gradient, learning_rate):
      self.theta -= learning_rate*gradient

    def predict_prob(self, batch_X, inverse_data_covariance = []):
      self.__initialize_theta(batch_X)
      batch_X = self.__update_batch(batch_X) 
      if len(inverse_data_covariance) == 0:     
        return self.__sigmoid(np.dot(batch_X, self.theta))
      else:
        ### adjust this using the inverse_data_covariance
        ### for each datapoint return the max 
        return self.__sigmoid(np.dot(batch_X, self.theta) - self.__inverse_covariance_norm(batch_X, inverse_data_covariance))


    def get_predictions(self, batch_X, threshold, inverse_data_covariance = []):
      prob_predictions = self.predict_prob(batch_X, inverse_data_covariance )
      thresholded_predictions= prob_predictions > threshold
      return thresholded_predictions

    def get_accuracy( self, batch_X, batch_y, threshold, inverse_data_covariance = []):
      thresholded_predictions = self.get_predictions(batch_X, threshold, inverse_data_covariance)
      return (thresholded_predictions == batch_y).mean()


    # def predict(self, batch_X, threshold):
    #   prob_predictions = self.predict_prob(batch_X)
    #   thresholded_predictions = prob_predictions > threshold
    #   return thresholded_predictions

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

