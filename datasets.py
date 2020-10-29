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


# @title Load Adult dataset
# num_samples = 3000

CATEGORICAL_COLUMNS = [
    'workclass', 'education', 'marital_status', 'occupation', 'relationship',
    'race', 'gender', 'native_country'
]
CONTINUOUS_COLUMNS = [
    'age', 'capital_gain', 'capital_loss', 'hours_per_week', 'education_num'
]
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]
LABEL_COLUMN = 'label'

PROTECTED_GROUPS = [
    'gender_Female', 'gender_Male', 'race_White', 'race_Black'
]

def get_adult_data():
 
  train_df_raw = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', 
                             names=COLUMNS, skipinitialspace=True)
  test_df_raw = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', 
                            names=COLUMNS, skipinitialspace=True, skiprows=1)

  # for column in train_df_raw.columns:
  #   if train_df_raw[column].dtype.name == 'category':
  #     categories_1 = set(train_df_raw[column].cat.categories)
  #     categories_2 = set(test_df_raw[column].cat.categories)
  #     categories = sorted(categories_1 | categories_2)
  #     train_df_raw[column].cat.set_categories(categories, inplace=True)
  #     test_df_raw[column].cat.set_categories(categories, inplace=True)

  # train_df_raw.dropna(inplace=True)
  # test_df_raw.dropna(inplace=True)

  train_df_raw[LABEL_COLUMN] = (
      train_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
  test_df_raw[LABEL_COLUMN] = (
      test_df_raw['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
  # Preprocessing Features
  pd.options.mode.chained_assignment = None  # default='warn'

  # Functions for preprocessing categorical and continuous columns.
  def binarize_categorical_columns(input_train_df,
                                   input_test_df,
                                   categorical_columns=[]):

    def fix_columns(input_train_df, input_test_df):
      test_df_missing_cols = set(input_train_df.columns) - set(
          input_test_df.columns)
      for c in test_df_missing_cols:
        input_test_df[c] = 0
      train_df_missing_cols = set(input_test_df.columns) - set(
          input_train_df.columns)
      for c in train_df_missing_cols:
        input_train_df[c] = 0
      input_train_df = input_train_df[input_test_df.columns]
      return input_train_df, input_test_df

    # Binarize categorical columns.
    binarized_train_df = pd.get_dummies(
        input_train_df, columns=categorical_columns)
    binarized_test_df = pd.get_dummies(
        input_test_df, columns=categorical_columns)
    # Make sure the train and test dataframes have the same binarized columns.
    fixed_train_df, fixed_test_df = fix_columns(binarized_train_df,
                                                binarized_test_df)
    return fixed_train_df, fixed_test_df
  
  def bucketize_continuous_column(input_train_df,
                                  input_test_df,
                                  continuous_column_name,
                                  num_quantiles=None,
                                  bins=None):
    assert (num_quantiles is None or bins is None)
    if num_quantiles is not None:
      train_quantized, bins_quantized = pd.qcut(
          input_train_df[continuous_column_name],
          num_quantiles,
          retbins=True,
          labels=False)
      input_train_df[continuous_column_name] = pd.cut(
          input_train_df[continuous_column_name], bins_quantized, labels=False)
      input_test_df[continuous_column_name] = pd.cut(
          input_test_df[continuous_column_name], bins_quantized, labels=False)
    elif bins is not None:
      input_train_df[continuous_column_name] = pd.cut(
          input_train_df[continuous_column_name], bins, labels=False)
      input_test_df[continuous_column_name] = pd.cut(
          input_test_df[continuous_column_name], bins, labels=False)

  # Filter out all columns except the ones specified.
  train_df = train_df_raw[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS +
                          [LABEL_COLUMN]]
  test_df = test_df_raw[CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS +
                        [LABEL_COLUMN]]
  # Bucketize continuous columns.
  bucketize_continuous_column(train_df, test_df, 'age', num_quantiles=4)
  bucketize_continuous_column(
      train_df, test_df, 'capital_gain', bins=[-1, 1, 4000, 10000, 100000])
  bucketize_continuous_column(
      train_df, test_df, 'capital_loss', bins=[-1, 1, 1800, 1950, 4500])
  bucketize_continuous_column(
      train_df, test_df, 'hours_per_week', bins=[0, 39, 41, 50, 100])
  bucketize_continuous_column(
      train_df, test_df, 'education_num', bins=[0, 8, 9, 11, 16])
  train_df, test_df = binarize_categorical_columns(
      train_df,
      test_df,
      categorical_columns=CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS)
  feature_names = list(train_df.keys())
  feature_names.remove(LABEL_COLUMN)
  num_features = len(feature_names)
  return train_df, test_df, feature_names


train_df, test_df, feature_names = get_adult_data()
# train_df = train_df.sample(num_samples)
# test_df = test_df.sample(int(num_samples/2))

X_train_adult_df = train_df[feature_names]
y_train_adult_df = train_df[LABEL_COLUMN]
X_test_adult_df = test_df[feature_names]
y_test_adult_df = test_df[LABEL_COLUMN]

X_train_adult = np.array(train_df[feature_names])
y_train_adult = np.array(train_df[LABEL_COLUMN])
X_test_adult = np.array(test_df[feature_names])
y_test_adult = np.array(test_df[LABEL_COLUMN])

protected_train_adult = [np.array(train_df[g]) for g in PROTECTED_GROUPS]
protected_test_adult = [np.array(test_df[g]) for g in PROTECTED_GROUPS]

def get_protected_dataframes(X_df, y_df, protected_groups):
  protected_dataframes = []
  for g in protected_groups:
    X_protected = X_df[X_df[g] == 1]
    y_protected = y_df[X_df[g] == 1]
    protected_dataframes.append((X_protected, y_protected))
  return protected_dataframes

all_data_dataframes_train = (X_train_adult_df, y_train_adult_df)
all_data_dataframes_test = (X_test_adult_df, y_test_adult_df)
protected_dataframes_train = get_protected_dataframes(X_train_adult_df, y_train_adult_df, PROTECTED_GROUPS)
protected_dataframes_test = get_protected_dataframes(X_test_adult_df, y_test_adult_df, PROTECTED_GROUPS)
print(np.shape(X_train_adult))
print(np.shape(X_test_adult))

def get_adult_disjoint(protected):
  combos = [(0,2), (0,3), (1,2), (1,3)]
  new_protected = [np.where(np.logical_and(protected[a], protected[b]), 1, 0) for a,b in combos]
  return new_protected

protected_train_adult_disjoint = get_adult_disjoint(protected_train_adult) 
protected_test_adult_disjoint = get_adult_disjoint(protected_test_adult)
PROTECTED_GROUPS = [
    'Female_White', 'Female_Black', 'Male_White', 'Male_Black'                
]



#@title Data utilities
class DataSet:
  def __init__(self, dataset, labels, num_classes = 2):
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
      X = self.dataset.sample(batch_size, random_state = self.random_state).values
      Y = self.labels.sample(batch_size, random_state = self.random_state).values
    # Y_one_hot = np.zeros((Y.shape[0], self.num_classes))
    # for i in range(self.num_classes):
    #   Y_one_hot[:, i] = (Y == i)*1.0
    self.random_state += 1

    return (X,Y)



class MixtureGaussianDataset:
  def __init__(self, means, 
               variances, 
               probabilities, 
               theta_stars, 
               num_classes=2, 
               max_batch_size = 10000, 
               kernel = lambda a,b : np.dot(a,b)):
    self.means = means
    self.variances = variances
    self.probabilities = probabilities
    self.num_classes = num_classes
    self.theta_stars = theta_stars
    self.cummulative_probabilities = np.zeros(len(probabilities))
    cum_prob = 0
    for i,prob in enumerate(self.probabilities):
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
      while index <= len(self.cummulative_probabilities)-1:
        if val < self.cummulative_probabilities[index]:
          break
        index += 1

      x = np.random.multivariate_normal(self.means[index], np.eye(self.dimension)*variances[index])
      logit = self.kernel(x, self.theta_stars[index])
      y_val = 1 / (1 + np.exp(-logit))
      y = (np.random.random() >= y_val)*1.0
      X.append(x)
      Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return (X,Y)





class SVMDataset:
  def __init__(self, means, 
               variances, 
               probabilities, 
               class_list_per_center,
               num_classes=2, 
               max_batch_size = 10000):
    self.means = means
    self.variances = variances
    self.probabilities = probabilities
    self.num_classes = num_classes
    self.class_list_per_center = class_list_per_center
    self.cummulative_probabilities = np.zeros(len(probabilities))
    cum_prob = 0
    for i,prob in enumerate(self.probabilities):
      cum_prob += prob
      self.cummulative_probabilities[i] = cum_prob
    self.max_batch_size = max_batch_size
    self.num_groups = len(self.means)
    self.dim = self.means[0].shape[0]

  def get_batch(self, batch_size, verbose = False):
    batch_size = min(batch_size, self.max_batch_size)
    X = []
    Y = []
    indices = []
    for _ in range(batch_size):
      val = np.random.random()
      index = 0  
      while index <= len(self.cummulative_probabilities)-1:
        if val < self.cummulative_probabilities[index]:
          break
        index += 1
      
      x = np.random.multivariate_normal(self.means[index], np.eye(self.dim)*variances[index])
      y = self.class_list_per_center[index]
      X.append(x)
      Y.append(y)
      indices.append(index)
    X = np.array(X)
    Y = np.array(Y)
    indices = np.array(indices)
    if verbose:
      return (X, Y, indices)
    else:
      return (X,Y)
  
  def plot(self, batch_size, model = None, names = []):
    if names == []:
      names = ["" for _ in range(self.num_groups)]
    if self.dim != 2:
      print("Unable to plot the dataset")
    else:
      colors = ["blue", "red", "green", "yellow", "black", "orange", "purple", "violet", "gray"]
      (X, Y, indices) = self.get_batch( batch_size, verbose = True)
      #print("xvals ", X, "yvals ", Y)
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
            
        plt.plot(X_filtered_0, X_filtered_1, 'o', color = colors[i]  , label = "{} {}".format(self.class_list_per_center[i], names[i]) )
      if model != None:
          ## Plot line
          model.plot(min_x, max_x)
      plt.grid(True)
      plt.legend(loc = "lower right")
    



def get_batches(protected_datasets, global_dataset, batch_size):
  global_batch = global_dataset.get_batch(batch_size)

  protected_batches = [protected_dataset.get_batch(batch_size) for protected_dataset in protected_datasets]
  return global_batch, protected_batches


def get_dataset(dataset):

  if dataset == "Mixture":
    PROTECTED_GROUPS = ["A", "B", "C", "D"]
    d = 20
    means = [ -10*np.arange(d)/np.linalg.norm(np.ones(d)), np.zeros(d),  10*np.arange(d)/np.linalg.norm(np.arange(d)), np.ones(d)/np.linalg.norm(np.ones(d))]
    variances = [.4, .41, .41, .41]
    theta_stars = [np.zeros(d),np.zeros(d), np.zeros(d), np.zeros(d)]
    probabilities = [ .3, .1, .5, .1 ]
    kernel = lambda a,b : .1*np.dot(a-b, a-b ) - 1

    protected_datasets_train = [MixtureGaussianDataset([means[i]], [variances[i]], [1], [theta_stars[i]], kernel = kernel) for i in range(len(PROTECTED_GROUPS))]
    protected_datasets_test = [MixtureGaussianDataset([means[i]], [variances[i]], [1], [theta_stars[i]], kernel = kernel) for i in range(len(PROTECTED_GROUPS))]

    train_dataset = MixtureGaussianDataset(means, variances, probabilities, theta_stars, kernel = kernel)
    test_dataset = MixtureGaussianDataset(means, variances, probabilities, theta_stars, kernel = kernel)
  elif dataset == "Adult":
    PROTECTED_GROUPS = [
        'Female_White', 'Female_Black', 'Male_White', 'Male_Black'                
    ]
    protected_datasets_train = [DataSet(x_vals, y_vals) for (x_vals, y_vals) in protected_dataframes_train]
    protected_datasets_test = [DataSet(x_vals, y_vals) for (x_vals, y_vals) in protected_dataframes_test]


    train_dataset = DataSet(X_train_adult_df, y_train_adult_df)
    test_dataset = DataSet(X_test_adult_df, y_test_adult_df)
  elif dataset == "MultiSVM":
    PROTECTED_GROUPS = ["A", "B", "C"]
    d = 2
    means = [ np.array([0, 5]), np.array([0, 0]), np.array([5, -2])]
    variances = [.5, .5, .5]
    probabilities = [ .3, .3, .4]
    class_list_per_center = [1, 0, 1]
    
    protected_datasets_train = [SVMDataset([means[i]], [variances[i]], [1],  [class_list_per_center[i]]) for i in range(len(PROTECTED_GROUPS))]
    protected_datasets_test = [SVMDataset([means[i]], [variances[i]], [1],  [class_list_per_center[i]]) for i in range(len(PROTECTED_GROUPS))]

    train_dataset = SVMDataset(means, variances, probabilities,  class_list_per_center)
    test_dataset = SVMDataset(means, variances, probabilities, class_list_per_center)


  elif dataset == "SVM":
    PROTECTED_GROUPS = ["A", "B"]
    d = 2
    means = [ -np.arange(d)/np.linalg.norm(np.arange(d)), np.ones(d)/np.linalg.norm(np.ones(d))]
    variances = [1, .1]
    probabilities = [ .5, .5]
    class_list_per_center = [0, 1]
    
    protected_datasets_train = [SVMDataset([means[i]], [variances[i]], [1],  [class_list_per_center[i]]) for i in range(len(PROTECTED_GROUPS))]
    protected_datasets_test = [SVMDataset([means[i]], [variances[i]], [1],  [class_list_per_center[i]]) for i in range(len(PROTECTED_GROUPS))]

    train_dataset = SVMDataset(means, variances, probabilities,  class_list_per_center)
    test_dataset = SVMDataset(means, variances, probabilities, class_list_per_center)
  else:
    raise ValueError("Unrecognized dataset")

  return protected_datasets_train, protected_datasets_test, train_dataset, test_dataset








