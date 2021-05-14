from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import pandas as pd
from six.moves import cPickle
from sklearn import model_selection

# @title Load Adult dataset


class AdultParams_nonuai(object):
    """A namespace for adult dataset constant parameters."""

    PATH_TRAIN = "./datasets/adult.data"  #'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    PATH_TEST = "./datasets/adult.test"  #'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

    CATEGORICAL_COLUMNS = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "native_country",
    ]

    CONTINUOUS_COLUMNS = [
        "age",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "education_num",
    ]

    COLUMNS = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income_bracket",
    ]

    LABEL_COLUMN = "label"

    CONTINUOUS_BUCKETS = tuple(
        {
            "capital-gain": (-1, 1, 4000, 10000, 100000),
            "capital-loss": (-1, 1, 1800, 1950, 4500),
            "hours-per-week": (0, 39, 41, 50, 100),
            "education-num": (0, 8, 9, 11, 16),
        }.items()
    )

    CONTINUOUS_QUANTILE_COUNTS = tuple(
        {
            "age": 4,
        }.items()
    )

    PROTECTED_GROUPS = (
        "gender_Female",
        "gender_Male",  #'race_Black', 'race_White',
    )

    JOINT_PROTECTED_GROUPS = (
        ("gender_Female", "race_Black"),
        ("gender_Male", "race_Black"),
        ("gender_Female", "race_White"),
        ("gender_Male", "race_White"),
        # ('gender_Female', 'race_Asian-Pac-Islander'),
        # ('gender_Male', 'race_Asian-Pac-Islander'),
        # ('gender_Female', 'race_Amer-Indian-Eskimo'),
        # ('gender_Male', 'race_Amer-Indian-Eskimo'),
        # ('gender_Female', 'race_Other'),
        # ('gender_Male', 'race_Other'),
    )


def get_data_nonuai(url, test=False):

    # out = requests.get(train_url)
    out = open(url)
    if test:
        data = [x.split(", ") for x in out.read().split("\n")[1:]]
    else:
        data = [x.split(", ") for x in out.read().split("\n")]
    df = pd.DataFrame(data, columns=AdultParams.COLUMNS)

    for col in AdultParams.CONTINUOUS_COLUMNS:
        df[col] = pd.to_numeric(df[col])

    for col in AdultParams.COLUMNS:
        df[col].replace("?", None, inplace=True)

    return df


# def read_and_preprocess_adult_data_nonuai(joint_protected_groups=True):
#
#     train_df_raw = get_data(AdultParams.PATH_TRAIN)
#     test_df_raw = get_data(AdultParams.PATH_TEST, test=True)
#
#     for column in train_df_raw.columns:
#         if train_df_raw[column].dtype.name == "category":
#             categories_1 = set(train_df_raw[column].cat.categories)
#             categories_2 = set(test_df_raw[column].cat.categories)
#             categories = sorted(categories_1 | categories_2)
#             train_df_raw[column].cat.set_categories(categories, inplace=True)
#             test_df_raw[column].cat.set_categories(categories, inplace=True)
#
#     train_df_raw.dropna(inplace=True)
#     test_df_raw.dropna(inplace=True)
#
#     train_df_raw[AdultParams.LABEL_COLUMN] = (
#         train_df_raw["income_bracket"].apply(lambda x: ">50K" in x)
#     ).astype(int)
#     test_df_raw[AdultParams.LABEL_COLUMN] = (
#         test_df_raw["income_bracket"].apply(lambda x: ">50K" in x)
#     ).astype(int)
#
#     # Preprocessing Features
#     pd.options.mode.chained_assignment = None  # default='warn'
#
#     # Functions for preprocessing categorical and continuous columns.
#     def binarize_categorical_columns(
#         input_train_df, input_test_df, categorical_columns=[]
#     ):
#         def fix_columns(input_train_df, input_test_df):
#             test_df_missing_cols = set(input_train_df.columns) - set(
#                 input_test_df.columns
#             )
#             for c in test_df_missing_cols:
#                 input_test_df[c] = 0
#             train_df_missing_cols = set(input_test_df.columns) - set(
#                 input_train_df.columns
#             )
#             for c in train_df_missing_cols:
#                 input_train_df[c] = 0
#             input_train_df = input_train_df[input_test_df.columns]
#             return input_train_df, input_test_df
#
#         # Binarize categorical columns.
#         binarized_train_df = pd.get_dummies(input_train_df, columns=categorical_columns)
#         binarized_test_df = pd.get_dummies(input_test_df, columns=categorical_columns)
#         # Make sure the train and test dataframes have the same binarized columns.
#         fixed_train_df, fixed_test_df = fix_columns(
#             binarized_train_df, binarized_test_df
#         )
#         return fixed_train_df, fixed_test_df
#
#     def bucketize_continuous_column(
#         input_train_df,
#         input_test_df,
#         continuous_column_name,
#         num_quantiles=None,
#         bins=None,
#     ):
#         assert num_quantiles is None or bins is None
#         if num_quantiles is not None:
#             train_quantized, bins_quantized = pd.qcut(
#                 input_train_df[continuous_column_name],
#                 num_quantiles,
#                 retbins=True,
#                 labels=False,
#             )
#             input_train_df[continuous_column_name] = pd.cut(
#                 input_train_df[continuous_column_name], bins_quantized, labels=False
#             )
#             input_test_df[continuous_column_name] = pd.cut(
#                 input_test_df[continuous_column_name], bins_quantized, labels=False
#             )
#         elif bins is not None:
#             input_train_df[continuous_column_name] = pd.cut(
#                 input_train_df[continuous_column_name], bins, labels=False
#             )
#             input_test_df[continuous_column_name] = pd.cut(
#                 input_test_df[continuous_column_name], bins, labels=False
#             )
#
#     # Filter out all columns except the ones specified.
#     train_df = train_df_raw[
#         AdultParams.CATEGORICAL_COLUMNS
#         + AdultParams.CONTINUOUS_COLUMNS
#         + [AdultParams.LABEL_COLUMN]
#     ]
#     test_df = test_df_raw[
#         AdultParams.CATEGORICAL_COLUMNS
#         + AdultParams.CONTINUOUS_COLUMNS
#         + [AdultParams.LABEL_COLUMN]
#     ]
#
#     # Bucketize continuous columns.
#     bucketize_continuous_column(train_df, test_df, "age", num_quantiles=4)
#     bucketize_continuous_column(
#         train_df, test_df, "capital_gain", bins=[-1, 1, 4000, 10000, 100000]
#     )
#     bucketize_continuous_column(
#         train_df, test_df, "capital_loss", bins=[-1, 1, 1800, 1950, 4500]
#     )
#     bucketize_continuous_column(
#         train_df, test_df, "hours_per_week", bins=[0, 39, 41, 50, 100]
#     )
#     bucketize_continuous_column(
#         train_df, test_df, "education_num", bins=[0, 8, 9, 11, 16]
#     )
#     train_df, test_df = binarize_categorical_columns(
#         train_df,
#         test_df,
#         categorical_columns=AdultParams.CATEGORICAL_COLUMNS
#         + AdultParams.CONTINUOUS_COLUMNS,
#     )
#     feature_names = list(train_df.keys())
#     feature_names.remove(AdultParams.LABEL_COLUMN)
#     num_features = len(feature_names)
#
#     return train_df, test_df, feature_names


_COLUMNS = (
    "age",
    "workclass",
    "final-weight",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
)
_CATEGORICAL_COLUMNS = (
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "race",
    "relationship",
    "sex",
    "native-country",
    "income",
)


def _read_data(name):

    with open(name) as data_file:
        # with resources.GetResourceAsFile(os.path.join(_DATA_PATH, name)) as data_file:
        data = pd.read_csv(
            data_file,
            header=None,
            index_col=False,
            names=_COLUMNS,
            skipinitialspace=True,
            na_values="?",
        )
        for categorical in _CATEGORICAL_COLUMNS:
            data[categorical] = data[categorical].astype("category")
    return data


def _combine_category_coding(df_1, df_2):
    """Combines the categories between dataframes df_1 and df_2.

    This is used to ensure that training and test data use the same category
    coding, so that the one-hot vectors representing the values are compatible
    between training and test data.

    Args:
      df_1: Pandas DataFrame.
      df_2: Pandas DataFrame. Must have the same columns as df_1.
    """
    for column in df_1.columns:
        if df_1[column].dtype.name == "category":
            categories_1 = set(df_1[column].cat.categories)
            categories_2 = set(df_2[column].cat.categories)
            categories = sorted(categories_1 | categories_2)
            df_1[column].cat.set_categories(categories, inplace=True)
            df_2[column].cat.set_categories(categories, inplace=True)


def read_all_data(remove_missing=True):
    """Return (train, test) dataframes, optionally removing incomplete rows."""
    train_data = _read_data("./datasets/adult.data")
    test_data = _read_data("./datasets/adult.test")
    _combine_category_coding(train_data, test_data)
    if remove_missing:
        train_data = train_data.dropna()
        test_data = test_data.dropna()
    return train_data, test_data


class AdultParams(object):
    """A namespace for adult dataset constant parameters."""

    CATEGORICAL_COLUMNS = (
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    )

    CONTINUOUS_COLUMNS = (
        "age",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "education-num",
    )

    LABEL_COLUMN = "label"

    CONTINUOUS_BUCKETS = tuple(
        {
            "capital-gain": (-1, 1, 4000, 10000, 100000),
            "capital-loss": (-1, 1, 1800, 1950, 4500),
            "hours-per-week": (0, 39, 41, 50, 100),
            "education-num": (0, 8, 9, 11, 16),
        }.items()
    )

    CONTINUOUS_QUANTILE_COUNTS = tuple(
        {
            "age": 4,
        }.items()
    )

    PROTECTED_GROUPS = (
        "sex_Female",
        "sex_Male",  # 'race_White', 'race_Black',
    )

    JOINT_PROTECTED_GROUPS = (
        ("sex_Female", "race_Black"),
        ("sex_Male", "race_Black"),
        ("sex_Female", "race_White"),
        ("sex_Male", "race_White"),
    )


def read_and_preprocess_adult_data_uai(
    remove_missing=True,
    continuous_column_buckets=AdultParams.CONTINUOUS_BUCKETS,
    continuous_column_quantile_counts=AdultParams.CONTINUOUS_QUANTILE_COUNTS,
):
    """Read and preprocess the UCB Adult dataset.

    Args:
      remove_missing: `remove_missing` arg to the `read_all_data` function in
          //l/d/r/causality/fairness/adult.py.
      continuous_column_buckets: a mapping from column names in the Adult dataset
          to tuples of bin edges, for creating indicator variables with
          `pandas.cut`. **Should be a tuple of pairs, not a dict!**
      continuous_column_quantile_counts: a mapping from column names in the adult
          dataset to a number of quantiles, for creating indicator variables with
          `pandas.qcut` and `pandas.cut`. **Should be a tuple of pairs, not a
          dict!**

    Returns:
        A 3-tuple with the following contents:
        [0]: A dataframe of training data.
        [1]: A dataframe of test data.
        [2]: A list of names of columns in the data to use as features.
        The training and test data include a binary column called 'label' which
        is usually the prediction objective for this dataset. The feature name
        list [2] does not contain 'label'.
    """
    # Load the data itself.
    train_df, test_df = read_all_data(remove_missing)

    # Add the binary label column to the datasets.
    for df in [train_df, test_df]:
        df[AdultParams.LABEL_COLUMN] = (
            df["income"].apply(lambda x: ">50K" in x).astype(int)
        )

    # Filter for the columns we want.
    desired_columns = list(
        (AdultParams.LABEL_COLUMN,)
        + AdultParams.CATEGORICAL_COLUMNS
        + AdultParams.CONTINUOUS_COLUMNS
    )
    train_df = train_df[desired_columns]
    test_df = test_df[desired_columns]

    # "Bucketise" some continuous columns. This means grouping their values
    # into predefined categorical bins on a variable-by-variable basis.
    def bucketise_continuous_column(dataframes, continuous_column_name, bins):
        for df in dataframes:
            # This line causes pandas to complain about incorrect behaviour relating
            # to assigning to a copy of a slice, but still yields the correct result.
            df[continuous_column_name] = pd.cut(
                df[continuous_column_name], bins, labels=False
            )

    # Or, "quantilise" some continuous columns. This means grouping their values
    # into categorical bins based on which of N quantile regions they occupy.
    # NOTE: only the first value in `dataframes` will be used to compute where
    # the quantiles are.
    def quantilise_continuous_column(dataframes, continuous_column_name, num_quantiles):
        _, bins = pd.qcut(
            dataframes[0][continuous_column_name],
            num_quantiles,
            retbins=True,
            labels=False,
        )
        bucketise_continuous_column(dataframes, continuous_column_name, bins)

    # (Actually do bucketisation and quantilisation.)
    for column_name, num_quantiles in continuous_column_quantile_counts:
        quantilise_continuous_column([train_df, test_df], column_name, num_quantiles)
    for column_name, buckets in continuous_column_buckets:
        bucketise_continuous_column([train_df, test_df], column_name, buckets)

    # "Binarise" categorical columns. This means spreading categorical columns
    # into groups of one-hot columns (or dummies/indicators), one per category.
    def binarise_categorical_columns(dataframes):
        """Binarise categorical columns."""
        # Convert categorical columns to dummies. By now, even our continous columns
        # should be categorical and up for binarisation.
        columns = list(AdultParams.CATEGORICAL_COLUMNS + AdultParams.CONTINUOUS_COLUMNS)
        dataframes = [pd.get_dummies(df, columns=columns) for df in dataframes]
        # Make sure all dataframes have the same columns---some dummy columns may be
        # missing from some dataframes if certain categorical values are not present
        # in those dataframes. First, identify columns in all of the dataframes:
        all_columns = set.union(*[set(df.columns) for df in dataframes])
        # For each dataframe, add missing columns and fill them with zeros.
        return [df.reindex(columns=all_columns, fill_value=0) for df in dataframes]

    train_df, test_df = binarise_categorical_columns([train_df, test_df])

    # Sort columns so that we can reuse coefficients between runs.
    train_df.reindex(columns=train_df.columns.sort_values())
    test_df.reindex(columns=test_df.columns.sort_values())

    # Collect names of the data to use as features.
    feature_names = list(train_df.keys())
    feature_names.remove(AdultParams.LABEL_COLUMN)

    # Lose incomplete rows.
    if remove_missing:
        train_df = train_df.dropna()
        test_df = test_df.dropna()

    # All done!
    return train_df, test_df, feature_names


def collect_adult_protected_dataframes(
    dataframe, protected_groups=AdultParams.PROTECTED_GROUPS
):
    """Collect data in `dataframe` with matching binary attributes.

    Args:
      dataframe: Dataframe as loaded by e.g. `read_and_preprocess_adult_data`.
      protected_groups: A sequence of strings or string collections controlling
          the data that appears in the corresponding location in the return value.
          * If the n'th element is a string, then the n'th element of the return
            value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[protected_groups[n]] == 1`.
          * If the n'th element is a collection, then the n'th element of the
            return value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[k] == 1` for all `k` in
            `protected_groups[n]`.

    Returns:
      A list of filtered dataframes as described above.
    """
    return _collect_protected_dataframes(dataframe, protected_groups)


def collect_adult_protected_xy(
    x_protected, y_protected, protected_groups=AdultParams.PROTECTED_GROUPS
):
    """Collect data in `dataframe` with matching binary attributes.

    Args:
      dataframe: Dataframe as loaded by e.g. `read_and_preprocess_crime_data`.
      protected_groups: A sequence of strings or string collections controlling
          the data that appears in the corresponding location in the return value.
          * If the n'th element is a string, then the n'th element of the return
            value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[protected_groups[n]] == 1`.
          * If the n'th element is a collection, then the n'th element of the
            return value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[k] == 1` for all `k` in
            `protected_groups[n]`.

    Returns:
      A list of filtered dataframes as described above.
    """
    return _collect_protected_xy(x_protected, y_protected, protected_groups)


### "Bank" dataset (https://youtu.be/0A64PjHN-WA) ###


class BankParams(object):
    """A namespace for "Bank" dataset constant parameters."""

    PATH = "./datasets/bank-additional-full.csv"

    CATEGORICAL_COLUMNS = (
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "day_of_week",
        "poutcome",
        "y",
    )

    CONTINUOUS_COLUMNS = (
        "campaign",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    )

    LABEL_COLUMN = "y_yes"

    CONTINUOUS_PERCENTILE_BUCKETS = tuple(
        {
            # NOTE: ['age'] should have one fewer item than PROTECTED_GROUPS:
            "age": (20.0, 40.0, 60.0, 80.0),
            "duration": (20.0, 40.0, 60.0, 80.0),
        }.items()
    )

    COLUMNS_TO_SUBTRACT_MEAN_FROM = (
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    )

    # NOTE: Should have one more item than CONTINUOUS_PERCENTILE_BUCKETS['age'].
    PROTECTED_GROUPS = ("age_0", "age_1", "age_2", "age_3", "age_4")


def read_and_preprocess_bank_data(
    continuous_percentile_buckets=BankParams.CONTINUOUS_PERCENTILE_BUCKETS,
    subtract_mean_from=BankParams.COLUMNS_TO_SUBTRACT_MEAN_FROM,
    test_set_fraction=0.2,
    rng_seed=42,
):
    """Read and preprocess the "Bank" dataset.

    Args:
      continuous_percentile_buckets: a mapping from column names in the Bank
          dataset to tuples of percentiles, for creating indicator variables with
          `pandas.cut` (once the percentile values are found). Percentile edges
          for 0% and 100% will be added automatically, so don't list those.
          **Should be a tuple of pairs, not a dict!**
      subtract_mean_from: a collection of columns for which we subtract the mean
          value from all values in the column.
      test_set_fraction: how much of the loaded data to put into the test set
          (the remainder going into the training set). This partitioning is
          performed randomly each time.
      rng_seed: Random number generator seed to use when partitioning the data
          into training and test sets. Specify None to use a random value
          taken from /dev/urandom. NOTE: there is a seed specified by default!

    Returns:
        A 3-tuple with the following contents:
        [0]: A dataframe of training data.
        [1]: A dataframe of test data.
        [2]: A list of names of columns in the data to use as features.
        The training and test data include a binary column called 'y_yes' which
        is usually the prediction objective for this dataset. The feature name
        list [2] does not contain 'y_yes'.
    """
    # Load the data itself.
    with open(BankParams.PATH) as f:
        dataframe = pd.read_csv(f, sep=";")

    # "Percentilise" some continuous columns. This means grouping their values
    # into categorical bins by percentile.
    def percentilise_continuous_column(df, continuous_column_name, percentiles):
        bin_edges = (
            [df[continuous_column_name].min() - 1.0]
            + list(np.percentile(df[continuous_column_name], percentiles))
            + [df[continuous_column_name].max() + 1.0]
        )
        df[continuous_column_name] = pd.cut(
            df[continuous_column_name], bin_edges, labels=False
        )

    for column_name, percentiles in continuous_percentile_buckets:
        percentilise_continuous_column(dataframe, column_name, percentiles)

    # "Binarise" categorical columns. This means spreading categorical columns
    # into groups of one-hot columns (or dummies/indicators), one per category.
    def binarise_categorical_columns(df):
        return pd.get_dummies(
            df,
            columns=list(BankParams.CATEGORICAL_COLUMNS)
            + [k for k, _ in continuous_percentile_buckets],
        )

    dataframe = binarise_categorical_columns(dataframe)

    # "Fill" the duration columns: that is, if there is a 1 in the column
    # "durationD.0" at a particular row, set the same row in all columns
    # "durationF.0" where F < D to 1. This operation only makes sense if duration
    # is in `continuous_percentile_buckets`.
    if "duration" in [k for k, _ in continuous_percentile_buckets]:
        buckets = dict(continuous_percentile_buckets)["duration"]
        to_fill = ["duration_{}".format(i) for i in range(1 + len(buckets))]
        for i in range(len(to_fill) - 1):
            dataframe[to_fill[i]] = dataframe[to_fill[i:]].max(axis=1)

    # Subtract the mean from certain columns.
    for c in subtract_mean_from:
        dataframe[c] = dataframe[c] - np.mean(dataframe[c])

    # Obtain training and test data.
    n_samples = len(dataframe)
    n_test = int(round(test_set_fraction * n_samples))

    rng = np.random.RandomState(rng_seed)
    indices = rng.permutation(n_samples)
    indices_test = indices[:n_test]

    test_set_mask = np.zeros(n_samples, dtype=bool)
    test_set_mask[indices_test] = True
    train_df = dataframe[~test_set_mask]
    test_df = dataframe[test_set_mask]

    # Collect names of the data to use as features.
    feature_names = list(dataframe.keys())
    feature_names.remove(BankParams.LABEL_COLUMN)
    # For some reason we don't use month features.
    feature_names = [fn for fn in feature_names if not fn.startswith("month")]
    # Or "pdays" features.
    feature_names = [fn for fn in feature_names if not fn.startswith("pdays")]
    # And keeping around "y_no" would be cheating!
    feature_names = [fn for fn in feature_names if not fn.startswith("y_")]

    # All done!
    return train_df, test_df, feature_names


def collect_bank_protected_dataframes(
    dataframe, protected_groups=BankParams.PROTECTED_GROUPS
):
    """Collect data in `dataframe` with matching binary attributes.

    Args:
      dataframe: Dataframe as loaded by e.g. `read_and_preprocess_bank_data`.
      protected_groups: A sequence of strings or string collections controlling
          the data that appears in the corresponding location in the return value.
          * If the n'th element is a string, then the n'th element of the return
            value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[protected_groups[n]] == 1`.
          * If the n'th element is a collection, then the n'th element of the
            return value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[k] == 1` for all `k` in
            `protected_groups[n]`.

    Returns:
      A list of filtered dataframes as described above.
    """
    return _collect_protected_dataframes(dataframe, protected_groups)


def collect_bank_protected_xy(
    x_protected, y_protected, protected_groups=BankParams.PROTECTED_GROUPS
):
    """Collect data in `dataframe` with matching binary attributes.

    Args:
      dataframe: Dataframe as loaded by e.g. `read_and_preprocess_crime_data`.
      protected_groups: A sequence of strings or string collections controlling
          the data that appears in the corresponding location in the return value.
          * If the n'th element is a string, then the n'th element of the return
            value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[protected_groups[n]] == 1`.
          * If the n'th element is a collection, then the n'th element of the
            return value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[k] == 1` for all `k` in
            `protected_groups[n]`.

    Returns:
      A list of filtered dataframes as described above.
    """
    return _collect_protected_xy(x_protected, y_protected, protected_groups)


### "Crime" dataset ###


class CrimeParams(object):
    """A namespace for "Crime" dataset constant parameters."""

    PATH_TRAIN = "./datasets/crime_train.csv"

    PATH_VALIDATION = "./datasets/crime_val.csv"

    PATH_TEST = "./datasets/crime_test.csv"

    LABEL_COLUMN = "label"

    REGRESSION_TARGET_COLUMN = "ViolentCrimesPerPop"

    EXCLUDED_COLUMNS = ("state", "county", "community", "communityname")

    PROTECTED_GROUPS = (
        "racepctblack_cat_low",
        "racepctblack_cat_high",
        "racePctAsian_cat_low",
        "racePctAsian_cat_high",
        "racePctWhite_cat_low",
        "racePctWhite_cat_high",
        "racePctHisp_cat_low",
        "racePctHisp_cat_high",
    )


def read_and_preprocess_crime_data():
    """Read and preprocess the "Crime" dataset.

    Returns:
        A 3-tuple with the following contents:
        [0]: A dataframe of training data.
        [1]: A dataframe of test data.
        [2]: A list of names of columns in the data to use as features.
        The training and test data include a binary column called 'label' which
        is usually the prediction objective for this dataset. The feature name
        list [2] does not contain 'label', nor 'ViolentCrimesPerPop' (a
        regression objective sometimes used for this dataset).
    """

    df_train = pd.read_csv(CrimeParams.PATH_TRAIN, delimiter=",")
    df_validation = pd.read_csv(CrimeParams.PATH_VALIDATION, delimiter=",")
    df_test = pd.read_csv(CrimeParams.PATH_TEST, delimiter=",")
    """
  # Load the data itself.
  with gfile.Open(CrimeParams.CNS_PATH_TRAIN) as f:
    df_train = pd.read_csv(f)
  with gfile.Open(CrimeParams.CNS_PATH_VALIDATION) as f:
    df_validation = pd.read_csv(f)
  with gfile.Open(CrimeParams.CNS_PATH_TEST) as f:
    df_test = pd.read_csv(f)"""

    # Get names of columns that should be used as features.
    feature_names = [
        c
        for c in df_train.keys()  # pylint: disable=g-complex-comprehension
        if c
        not in list(CrimeParams.EXCLUDED_COLUMNS)
        + [CrimeParams.LABEL_COLUMN, CrimeParams.REGRESSION_TARGET_COLUMN]
    ]

    # Replace missing feature values with per-column training set means.
    for column in feature_names:
        train_mean = df_train[column].mean()
        df_train[column].fillna(train_mean, inplace=True)
        df_validation[column].fillna(train_mean, inplace=True)
        df_test[column].fillna(train_mean, inplace=True)

    # Just use the validation data for training.
    df_train = pd.concat([df_train, df_validation])

    # All done!
    return df_train, df_test, feature_names


def linear_regression_read_and_preprocess_crime_data():
    """Read and preprocess the "Crime" dataset for linear regression.

    Unlike `read_and_preprocess_crime_data`, we retain the column
    `ViolentCrimesPerPop` (the linear regression target) and drop the column
    `label` (the classification target).

    Returns:
        A 3-tuple with the following contents:
        [0]: A dataframe of training data.
        [1]: A dataframe of test data.
        [2]: A list of names of columns in the data to use as features.
        The training and test data include a numerical column called
        'ViolentCrimesPerPop' which is usually the prediction objective for this
        dataset. The feature name list [2] does not contain 'ViolentCrimesPerPop',
        nor 'label' (a classification objective sometimes used for this dataset).
    """

    df_train = pd.read_csv(CrimeParams.PATH_TRAIN)
    df_validation = pd.read_csv(CrimeParams.PATH_VALIDATION)
    df_test = pd.read_csv(CrimeParams.PATH_TEST)

    """# Load the data itself.
  with gfile.Open(CrimeParams.CNS_PATH_TRAIN) as f:
    df_train = pd.read_csv(f)
  with gfile.Open(CrimeParams.CNS_PATH_VALIDATION) as f:
    df_validation = pd.read_csv(f)
  with gfile.Open(CrimeParams.CNS_PATH_TEST) as f:
    df_test = pd.read_csv(f)"""

    # Get names of columns that should be used as features.
    feature_names = [
        c
        for c in df_train.keys()  # pylint: disable=g-complex-comprehension
        if c
        not in list(CrimeParams.EXCLUDED_COLUMNS)
        + [CrimeParams.LABEL_COLUMN, CrimeParams.REGRESSION_TARGET_COLUMN]
    ]

    # Replace missing feature values with per-column training set means.
    for column in feature_names:
        train_mean = df_train[column].mean()
        df_train[column].fillna(train_mean, inplace=True)
        df_validation[column].fillna(train_mean, inplace=True)
        df_test[column].fillna(train_mean, inplace=True)

    # Just use the validation data for training.
    df_train = pd.concat([df_train, df_validation])

    # All done!
    return df_train, df_test, feature_names


def collect_crime_protected_dataframes(
    dataframe, protected_groups=CrimeParams.PROTECTED_GROUPS
):
    """Collect data in `dataframe` with matching binary attributes.

    Args:
      dataframe: Dataframe as loaded by e.g. `read_and_preprocess_crime_data`.
      protected_groups: A sequence of strings or string collections controlling
          the data that appears in the corresponding location in the return value.
          * If the n'th element is a string, then the n'th element of the return
            value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[protected_groups[n]] == 1`.
          * If the n'th element is a collection, then the n'th element of the
            return value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[k] == 1` for all `k` in
            `protected_groups[n]`.

    Returns:
      A list of filtered dataframes as described above.
    """
    return _collect_protected_dataframes(dataframe, protected_groups)


def collect_crime_protected_xy(
    x_protected, y_protected, protected_groups=CrimeParams.PROTECTED_GROUPS
):
    """Collect data in `dataframe` with matching binary attributes.

    Args:
      dataframe: Dataframe as loaded by e.g. `read_and_preprocess_crime_data`.
      protected_groups: A sequence of strings or string collections controlling
          the data that appears in the corresponding location in the return value.
          * If the n'th element is a string, then the n'th element of the return
            value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[protected_groups[n]] == 1`.
          * If the n'th element is a collection, then the n'th element of the
            return value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[k] == 1` for all `k` in
            `protected_groups[n]`.

    Returns:
      A list of filtered dataframes as described above.
    """
    return _collect_protected_xy(x_protected, y_protected, protected_groups)


### "LSAC" (Law School Admission Council) dataset ###


class LsacParams(object):
    """A namespace for "LSAC" dataset constant parameters."""

    PATH = "./datasets/lsac.csv"

    REGRESSION_TARGET_COLUMN = "zgpa"

    REQUIRED_COLUMNS = (
        "female",
        "male",
        "race1",
    )

    NUMERICAL_COLUMNS = (
        "lsat",
        "DOB_yr",
        "fulltime",
        "decile1",
        "decile1b",
        "decile3",
    )

    CATEGORICAL_COLUMNS = (
        "race1",
        "cluster",
        "fam_inc",
    )

    PROTECTED_GROUPS = (
        "race1_white",
        "race1_non_white",
        "male",
        "female",
    )

    JOINT_PROTECTED_GROUPS = (
        ("race1_non_white", "female"),
        ("race1_non_white", "male"),
        ("race1_white", "female"),
        ("race1_white", "male"),
    )

    OMIT_COLUMNS = (
        # for just considering "white" and "non-white" :-/
        "race1_asian",
        "race1_black",
        "race1_hisp",
        "race1_other",
    )


def linear_regression_read_and_preprocess_lsac_data():
    """Read and preprocess the "LSAC" dataset.

    Returns:
        A 3-tuple with the following contents:
        [0]: A dataframe of training data.
        [1]: A dataframe of test data.
        [2]: A list of names of columns in the data to use as features.  The
        training and test data include a numerical column named at
        `LsacParams.REGRESSION_TARGET_COLUMN` which is usually the prediction
        objective for this dataset. The feature name list [2] does not contain
        this name.
    """

    data = pd.read_csv(LsacParams.PATH, delimiter=",")

    # with gfile.Open(LsacParams.CNS_PATH) as f:
    #   data = pd.read_csv(f, delimiter=',')

    # Add group attribute "female" (complement "male" column).
    female = 1 - data["male"]  # questionable...
    data = data.assign(female=female)

    # Drop rows where protected labels or "required columns" are missing.
    data = data[~data[LsacParams.REGRESSION_TARGET_COLUMN].isna()]
    for group in LsacParams.REQUIRED_COLUMNS:
        data = data[~data[group].isna()]

    # Shift DOB year by median (it has fewer missing values than age).
    dob_yr_median = data["DOB_yr"].median()
    data = data.assign(DOB_yr=data["DOB_yr"].subtract(dob_yr_median))

    # Helpers for assembling data into columns.

    # If any missing data (NaN) appears in `d`, replace it with 0.0, then add an
    # additional binary column indicating where the missing data was. I'm not sure
    # this approach will be all that beneficial to linear models; guess they could
    # learn to add the mean?
    def fill_missing_numerical(d):
        d = d.apply(float)
        missing_rows = d.isna()
        if missing_rows.any():
            d = d.fillna(0.0)
            missing_rows.apply(float)
            missing_rows.rename(d.name + "_is_missing", inplace=True)
            d = pd.concat((d, missing_rows), axis=1)
        return d

    # Convert categorical variable into dummy columns.
    def vectorise_categorical(d):
        d_onehot = pd.get_dummies(d)
        names = {s: "{}_{!s}".format(d.name, s) for s in d_onehot.columns}
        d_onehot.rename(columns=names, inplace=True)
        return d_onehot

    # Assemble columns with the features and groups that we want. (Note: some
    # members of this list can have multiple columns.
    columnses = [data[LsacParams.REGRESSION_TARGET_COLUMN]]
    for numcol in LsacParams.NUMERICAL_COLUMNS:
        columnses.append(fill_missing_numerical(data[numcol]))
    for catcol in LsacParams.CATEGORICAL_COLUMNS:
        columnses.append(vectorise_categorical(data[catcol]))

    # Add back male and female columns, which we don't vectorise like the other
    # categorical columns.
    columnses.append(data["male"])
    columnses.append(data["female"])

    data = pd.concat(columnses, axis=1)  # Glue columns together.

    # Add a non-white column derived from 'race1_white'.
    race1_non_white = 1 - data["race1_white"]
    data = data.assign(race1_non_white=race1_non_white)

    # Get rid of columns we mean to omit.
    data = data[[c for c in data.columns if c not in LsacParams.OMIT_COLUMNS]]

    # Sort columns so that we can reuse coefficients between runs.
    data = data.reindex(columns=data.columns.sort_values())
    data = data.reset_index(drop=True)

    # Collect names of data to use as features.
    feature_names = list(data.keys())
    feature_names.remove(LsacParams.REGRESSION_TARGET_COLUMN)

    # Split into training and test data.
    inds_train, inds_test = model_selection.train_test_split(
        np.arange(data.shape[0]), test_size=0.25, random_state=123
    )

    train_df = data.loc[inds_train].reset_index(drop=True)
    test_df = data.loc[inds_test].reset_index(drop=True)

    # Convert everything to floats.
    train_df = train_df.astype(np.float32)
    test_df = test_df.astype(np.float32)

    # All done!
    return train_df, test_df, feature_names


def collect_lsac_protected_dataframes(
    dataframe, protected_groups=LsacParams.PROTECTED_GROUPS
):
    """Collect data in `dataframe` with matching binary attributes.

    Args:
      dataframe: Dataframe as loaded by e.g. `read_and_preprocess_lsac_data`.
      protected_groups: A seuence of strings or string collections controlling
          the data that appears in the corresponding location in the return value.
          * If the n'th element is a string, then the n'th element of the return
            value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[protected_groups[n]] == 1`.
          * If the n'th element is a collection, then the n'th element of the
            return value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[k] == 1` for all `k` in
            `protected_groups[n]`.

    Returns:
      A list of filtered dataframes as described above.
    """
    return _collect_protected_dataframes(dataframe, protected_groups)


def collect_lsac_protected_xy(
    x_protected, y_protected, protected_groups=LsacParams.PROTECTED_GROUPS
):
    """Collect data in `dataframe` with matching binary attributes.

    Args:
      dataframe: Dataframe as loaded by e.g. `read_and_preprocess_crime_data`.
      protected_groups: A sequence of strings or string collections controlling
          the data that appears in the corresponding location in the return value.
          * If the n'th element is a string, then the n'th element of the return
            value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[protected_groups[n]] == 1`.
          * If the n'th element is a collection, then the n'th element of the
            return value will be a dataframe `df` containing only those rows of
            `dataframe` where `dataframe[k] == 1` for all `k` in
            `protected_groups[n]`.

    Returns:
      A list of filtered dataframes as described above.
    """
    return _collect_protected_xy(x_protected, y_protected, protected_groups)


### "German" dataset ###


class GermanParams(object):
    """A namespace for "German" dataset constant parameters."""

    PATH = "./datasets/statlog_german_credit.pkl"

    LABEL_COLUMN = "label"

    PROTECTED_THRESHOLDS = tuple(
        [
            ("feature_9", 30),  # Feature 9 is age.
        ]
    )


def read_and_preprocess_german_data(test_set_fraction=0.33, rng_seed=42):
    """Read and preprocess the "German" dataset.

    Args:
      test_set_fraction: how much of the loaded data to put into the test set
          (the remainder going into the training set). This partitioning is
          performed randomly each time.
      rng_seed: Random number generator seed to use when partitioning the data
          into training and test sets. Specify None to use a random value
          taken from /dev/urandom. NOTE: there is a seed specified by default!

    Returns:
        A 3-tuple with the following contents:
        [0]: A dataframe of training data.
        [1]: A dataframe of test data.
        [2]: A list of names of columns in the data to use as features.
        The training and test data include a binary column called 'label' which
        is usually the prediction objective for this dataset. The feature name
        list [2] does not contain 'label'.
    """
    # Load the data itself.
    with open(GermanParams.PATH, "rb") as f:
        raw_features = cPickle.load(f, encoding="latin1")
        raw_labels = cPickle.load(f, encoding="latin1")

    # Turn the data into a dataframe.
    raw_all_data = np.hstack([raw_features, np.expand_dims(raw_labels, axis=1)])
    feature_names = ["feature_{}".format(i) for i in range(raw_features.shape[1])]
    dataframe = pd.DataFrame(
        data=raw_all_data, columns=(feature_names + [GermanParams.LABEL_COLUMN])
    )

    # Split into training and test data.
    n_samples = len(dataframe)
    n_test = int(round(test_set_fraction * n_samples))

    rng = np.random.RandomState(rng_seed)
    indices = rng.permutation(n_samples)
    indices_test = indices[:n_test]

    test_set_mask = np.zeros(n_samples, dtype=bool)
    test_set_mask[indices_test] = True
    train_df = dataframe[~test_set_mask]
    test_df = dataframe[test_set_mask]

    # All done!
    return train_df, test_df, feature_names


def collect_german_protected_dataframes(
    dataframe, protected_thresholds=GermanParams.PROTECTED_THRESHOLDS
):
    """Portion out data in `dataframe` according to various thresholds.

    Args:
      dataframe: Dataframe as loaded by e.g. `read_and_preprocess_german_data`.
      protected_thresholds: a sequence of `(column name, threshold)` 2-tuples.
          Please **NOTE: NOT A DICT.** For each entry in protected_thresholds, the
          list returned by this function will have two dataframes, one where
          values in the named column are less than or equal to the threshold and
          one where those values exceed the threshold.

    Returns:
      A sequence of dataframes as described above.
    """
    result = []
    for feature, threshold in protected_thresholds:
        result.append(dataframe[dataframe[feature] <= threshold])
        result.append(dataframe[dataframe[feature] > threshold])
    return result


def collect_german_protected_xy(
    x_df, y_df, protected_thresholds=GermanParams.PROTECTED_THRESHOLDS
):
    """Portion out data in `dataframe` according to various thresholds.

    Args:
      dataframe: Dataframe as loaded by e.g. `read_and_preprocess_german_data`.
      protected_thresholds: a sequence of `(column name, threshold)` 2-tuples.
          Please **NOTE: NOT A DICT.** For each entry in protected_thresholds, the
          list returned by this function will have two dataframes, one where
          values in the named column are less than or equal to the threshold and
          one where those values exceed the threshold.

    Returns:
      A sequence of dataframes as described above.
    """
    result = []
    for feature, threshold in protected_thresholds:
        filtered_x = x_df[x_df[feature] <= threshold]
        filtered_y = y_df[x_df[feature] <= threshold]
        filtered2_x = x_df[x_df[feature] > threshold]
        filtered2_y = y_df[x_df[feature] > threshold]
        result.append((filtered_x, filtered_y))
        result.append((filtered2_x, filtered2_y))
    return result


# @title Utilities

### Utilities ###
def _collect_protected_dataframes(dataframe, protected_groups):
    """Implementation of `collect_*_protected_dataframes`."""
    # Turn singleton members of protected_groups into tuples. (Non-string members
    # are assumed to be sequences already.)
    protected_groups = [(g,) if isinstance(g, str) else g for g in protected_groups]

    # Perform filtering for each entry in protected_groups.
    result = []
    for grouping in protected_groups:
        filtered = dataframe
        for group in grouping:
            filtered = filtered[filtered[group] == 1]
        result.append(filtered)

    return result


def _collect_protected_xy(x_df, y_df, protected_groups):
    """Implementation of `collect_*_protected_dataframes`."""
    # Turn singleton members of protected_groups into tuples. (Non-string members
    # are assumed to be sequences already.)
    protected_groups = [(g,) if isinstance(g, str) else g for g in protected_groups]

    # Perform filtering for each entry in protected_groups.
    result = []
    for grouping in protected_groups:
        filtered_x = x_df
        filtered_y = y_df
        for group in grouping:
            filtered_y = filtered_y[filtered_x[group] == 1]
            filtered_x = filtered_x[filtered_x[group] == 1]
        result.append((filtered_x, filtered_y))

    return result


def _collect_protected_indicators(dataframe, protected_groups):
    """Implementation of `collect_*_protected_indicators`."""
    # Turn singleton members of protected_groups into tuples. (Non-string members
    # are assumed to be sequences already.)
    protected_groups = [(g,) if isinstance(g, str) else g for g in protected_groups]

    # Assemble indicators for each member of protected_groups.
    indicators = []
    for grouping in protected_groups:
        indicator = True
        for group in grouping:
            indicator &= dataframe[group] != 0.0
        indicators.append(indicator.astype(np.float32))

    return indicators
