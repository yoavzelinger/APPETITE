from pandas import DataFrame, Series, read_csv
from pandas.api.types import is_numeric_dtype
from random import seed, choices
from typing import Callable, Generator

from DataManagementTools.lazy_utils import lazy_product, SINGLE_ARGUMENT_EACH_GENERATOR

"""
This module is responsible for simulating concept drifts in a given dataset.
The module provides a generator that generates all possible concept drifts for a given list of features.
The concept drifts are simulated in the following ways:
    - Numeric features: The drift is simulated by adding a multiple of the standard deviation to all entries in the feature.
        you can control the drift size by changing the NUMERIC_DRIFT_SIZES tuple.
    - Categorical features: The drift is simulated by fixing a given value for a proportion of the data (which without the given value).
        you can control the drift proportion by changing the CATEGORICAL_PROPORTIONS tuple.

The module provides the following generators:
    - multiple_features_concept_drift_generator: Generate all possible concept drifts in a given list of features.
    - single_feature_concept_drift_generator: Generate all possible concept drifts in a given feature.
    
    * The functions are lazy and generate the drifts on the fly.
    
    Note that the original DataFrame is not changed, and the drifts are generated in new DataFrames\Series.

Example functions:
    - multiple_drifts_example: Generate all possible concept drifts in a given list of features.
    - single_drift_example: Generate all possible concept drifts in a given feature.

    All you need to do to run is to edit the constants in example_preparation:
        - DIRECTORY: The directory of the dataset.
        - FILE_PATH: The path of the dataset.
        - DRIFTING_FEATURES: The features to drift and their types (numeric or categorical). Can be left with None for automatic detection.

Good Luck!
"""

NUMERIC_DRIFT_SIZES = (
    -2, 
    -1, 
    -0.5, 
    0.5, 
    1, 
    2
)
CATEGORICAL_PROPORTIONS = (
    0.3, 
    0.5, 
    0.7, 
    0.9
)
RANDOM_STATE = 10

def _numeric_drift_generator(
        column: Series,
 ) -> Generator[tuple[Series, str], None, None]:
    """
    Generator for all type of concept drifts in a numeric feature.

    Parameters:
        column (Series): The input column.

    Return:
        Generator[tuple[Series, str], None, None]: A generator of all drifts in the feature and the description of the drift.
    """
    assert is_numeric_dtype(column)
    
    feature_std = column.std()

    #   Nested function
    def simulate_numeric_drift_of_size_k(
            k: int
     ) -> tuple[Series, str]:
        """
        Simulate concept drift in a specific numeric feature of size k.

        Parameters:
            k (int): The value to add to all entries in the specified column.

        Returns:
            tuple[Series, str]: The new Series with the concept drift and a description of the drift (in the format of "NumericFeature[feature;{+/-}kstd]").
        """
        return (column + k * feature_std, f"NumericFeature[{column.name};{'+' if k >= 0 else ''}{k}std]")
    
    #   Using it in iterations
    for k in NUMERIC_DRIFT_SIZES:
        yield simulate_numeric_drift_of_size_k(k)



def _categorical_drift_generator(
        column: Series
 ) -> Generator[tuple[Series, str], None, None]:
    """
    Simulate concept drift in a specific categorical feature.
    The drift is simulated in every feature value in every relevant proportion size.

    Parameters:
        column (Series): The input column.
    
    Returns:
        Generator[tuple[Series, str], None, None]: A generator of all drifts in the feature and the description of the drift.
    """
    assert not is_numeric_dtype(column)
    
    unique_values = column.unique()
    
    # Nested function
    def categorical_drift_in_value_generator(
            fixed_value: str
     ) -> list[tuple[Series, str]]:
        """
        Simulate concept drift in a specific value of a feature (for all proportions).

        Parameters:
            fixed_value (str): The value that the drift is fixed to.

        Returns:
            list[tuple[Series, str]]: A list of all drifts in the feature and the description of the drift.
        """
        assert fixed_value in unique_values

        #   Double nested function
        def simulate_categorical_drift_in_value_proportion(
                p: float
         ) -> tuple[Series, str]:
            """
            Simulate concept drift in a specific value of a feature for a given proportion.
            The Drift is done by fixing a given value for a proportion of the data.

            Parameters:
                p (float): the proportion of the samples (out of the remaining samples - do not contains the value) that the fixed value will be inserted.

            Returns:
                tuple[Series, str]: The new Series with the concept drift and a description of the drift (in the format of "CategoricalFeature[feature=value;p=proportion]").
            """

            drifted_column = column.copy()
            remaining_indices = column != fixed_value
            remaining_count = remaining_indices.values.sum()
            remaining_indices = column[remaining_indices].index.values
            seed(RANDOM_STATE)
            fixed_indices = choices(remaining_indices, k=int(remaining_count * p))
            drifted_column[fixed_indices] = fixed_value

            return (drifted_column, f"CategoricalFeature[{column.name}={fixed_value};p={str(p).replace('.',',')}]")
        
        #   Using the doubly nested function
        return (
            simulate_categorical_drift_in_value_proportion(p)
            for p in CATEGORICAL_PROPORTIONS
        )
    
    # Using the intermediate generator
    for feature_value in unique_values:
        for drifted_column in categorical_drift_in_value_generator(feature_value):
            yield drifted_column

def _get_feature_generator_function(
        column: Series,
        type: str = None
 ) -> Callable[[Series], Generator[tuple[Series, str], None, None]]:
    """
    Get the relevant drift generator function for a given feature.
    
    Parameters:
        column (Series): The input column.
        type (str): The type of the feature.
        
    Returns:
        Callable[[Series], Generator[tuple[Series, str], None, None]]: The relevant drift generator function for the feature.
    """
    if type:
        return _numeric_drift_generator if type == "numeric" else _categorical_drift_generator
    if is_numeric_dtype(column):
        return _numeric_drift_generator
    return _categorical_drift_generator

# The magic starts here
def multiple_features_concept_drift_generator(
        original_df: DataFrame, 
        drifting_features: dict[str, str]
 ) -> Generator[tuple[DataFrame, str], None, None]:
    """
    Generate all possible concept drifts in a given list of features.
    Parameters:
        original_df (DataFrame): The original DataFrame.
        drifting_features (dict[str, str]): The features to drift and their types.
        
    Returns:
        Generator[tuple[DataFrame, str], None, None]: A generator of all possible drifts in the features and the description of the drift.
    """
    # Get features concept drift generators
    features_columns = [original_df[feature] for feature in drifting_features]
    generator_functions = [_get_feature_generator_function(column, feature_type) if feature_type else _get_feature_generator_function(column) 
                           for column, feature_type in zip(features_columns, drifting_features.values())]

    # Get the cartesian product of all drifts
    cartesian_products = lazy_product(generator_functions, args_lists=features_columns, args_type=SINGLE_ARGUMENT_EACH_GENERATOR)
    for drifts in cartesian_products:
        drifted_df = original_df.copy()
        drift_description = original_df.attrs.get("name", "") + '_'
        for drifted_column, current_description in drifts:
            drifted_df[drifted_column.name] = drifted_column
            drift_description += '_' + current_description
        yield (drifted_df, drift_description)

def single_feature_concept_drift_generator(
        data: DataFrame | Series, 
        feature: str = "",
        feature_type: str = None
 ) -> Generator[tuple[DataFrame, str], None, None]:
    """
    Generate all possible concept drifts in a given feature.
    Parameters:
        data (DataFrame | Series): The original DataFrame or the column from the DataFrame.
        feature (str): The feature to drift.
        feature_type (str): The type of the feature.
        
    Returns:
        Generator[tuple[DataFrame, str], None, None]: A generator of all possible drifts in the feature and the description of the drift.
    """
    column, is_data_df = data, False
    if isinstance(data, DataFrame):
        assert feature is not None and feature in data.columns
        column, is_data_df = data[feature], True
    assert isinstance(column, Series)

    generator_function = _get_feature_generator_function(column, feature_type)
    for drifted_column, drift_description in generator_function(column):
        if not is_data_df:
            yield (drifted_column, drift_description)
        else:
            drifted_df = data.copy()
            drifted_df[feature] = drifted_column
            yield (drifted_df, data.attrs.get("name", "") + '__' + drift_description)

def example_preparation(
        single_drift: bool = False
 ) -> tuple[DataFrame , dict[str, str]]:
    DIRECTORY = "data\\Classification_Datasets"
    FILE_PATH = "white-clover.csv"
    DRIFTING_FEATURES = {
        # For single drift only the first feature matters
        "WhiteClover-91": "numeric",
        "strata": "categorical",
        "Weeds-94": None
    }

    df = read_csv(f"{DIRECTORY}\\{FILE_PATH}")
    df.attrs['name'] = FILE_PATH.split("\\")[-1].split(".")[0]
    if single_drift:
        return df, *list(DRIFTING_FEATURES.items())[0]
    return df, DRIFTING_FEATURES

def multiple_drifts_example() -> Generator[tuple[DataFrame, str], None, None]:
    df, drifting_features = example_preparation()
    for drifted_df, drift_description in multiple_features_concept_drift_generator(df, drifting_features):
        yield (drifted_df, drift_description)

def single_drift_example() -> Generator[tuple[DataFrame, str], None, None]:
    df, feature, feature_type = example_preparation(True)
    for drifted_df, drift_description in single_feature_concept_drift_generator(df, feature, feature_type):
        yield (drifted_df, drift_description)