import pandas as pd
import random
from typing import Callable, Generator

from YoavNewCode.DataHandler.lazy_utils import lazy_product, SINGLE_ARGUMENT_EACH_GENERATOR

FILE_PATHES = (
    "white-clover.csv",
)
DIRECTORY = "data\\Classification_Datasets"

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

def _numeric_drift_generator(
        column: pd.Series,
 ) -> Generator[tuple[pd.Series, str], None, None]:
    """
    Generator for all type of concept drifts in a numeric feature.

    Parameters:
        column (pd.Series): The input column.

    Return:
        Generator[tuple[pd.Series, str], None, None]: A generator of all drifts in the feature and the description of the drift.
    """
    assert pd.api.types.is_numeric_dtype(column)
    
    feature_std = column.std()

    #   Nested function
    def simulate_numeric_drift_of_size_k(
            k: int
     ) -> tuple[pd.Series, str]:
        """
        Simulate concept drift in a specific numeric feature of size k.

        Parameters:
            k (int): The value to add to all entries in the specified column.

        Returns:
            tuple[pd.Series, str]: The new Series with the concept drift and a description of the drift (in the format of "NumericFeature[feature{+/-}kstd]").
        """
        return (column + k * feature_std, f"NumericFeature[{column.name}{'+' if k >= 0 else ''}{k}std]")
    
    #   Using it in iterations
    for k in NUMERIC_DRIFT_SIZES:
        yield simulate_numeric_drift_of_size_k(k)



def _categorical_drift_generator(
        column: pd.Series
 ) -> Generator[tuple[pd.Series, str], None, None]:
    """
    Simulate concept drift in a specific categorical feature.
    The drift is simulated in every feature value in every relevant proportion size.

    Parameters:
        column (pd.Series): The input column.
    
    Returns:
        Generator[tuple[pd.Series, str], None, None]: A generator of all drifts in the feature and the description of the drift.
    """
    assert not pd.api.types.is_numeric_dtype(column)
    
    unique_values = column.unique()
    
    # Nested function
    def categorical_drift_in_value_generator(
            fixed_value: str
     ) -> list[tuple[pd.Series, str]]:
        """
        Simulate concept drift in a specific value of a feature (for all proportions).

        Parameters:
            fixed_value (str): The value that the drift is fixed to.

        Returns:
            list[tuple[pd.Series, str]]: A list of all drifts in the feature and the description of the drift.
        """
        assert fixed_value in unique_values

        #   Double nested function
        def simulate_categorical_drift_in_value_proportion(
                p: float
         ) -> tuple[pd.Series, str]:
            """
            Simulate concept drift in a specific value of a feature for a given proportion.
            The Drift is done by fixing a given value for a proportion of the data.

            Parameters:
                p (float): the proportion of the samples (out of the remaining samples - do not contains the value) that the fixed value will be inserted.

            Returns:
                tuple[pd.Series, str]: The new Series with the concept drift and a description of the drift (in the format of "CategoricalFeature[feature=value;p=proportion]").
            """

            drifted_column = column.copy()
            remaining_indices = column != fixed_value
            remaining_count = remaining_indices.values.sum()
            remaining_indices = column[remaining_indices].index.values
            random.seed(10)
            fixed_indicies = random.choices(remaining_indices, k=int(remaining_count * p))
            drifted_column[fixed_indicies] = fixed_value

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
        column: pd.Series,
        type: str = None
 ) -> Callable[[], Generator[tuple[pd.Series, str], pd.Series, None]]:
    """
    Get the relevant drift generator function for a given feature.
    
    Parameters:
        column (pd.Series): The input column.
        type (str): The type of the feature.
        
    Returns:
        Callable[[], Generator[pd.Series, pd.Series, None]]: The relevant drift generator function for the feature.
    """
    if type:
        return _numeric_drift_generator if type == "numeric" else _categorical_drift_generator
    if pd.api.types.is_numeric_dtype(column):
        return _numeric_drift_generator
    return _categorical_drift_generator

# Now the magic happens
def concept_drifts_generator(
        original_df: pd.DataFrame, 
        drifting_features: dict[str, str]
 ) -> Generator[tuple[pd.DataFrame, str], None, None]:
    """
    Generate all possible concept drifts in a given list of features.
    Parameters:
        original_df (pd.DataFrame): The original DataFrame.
        drifting_features (list[str]): List of features to drift.
        
    Returns:
        Generator[tuple[pd.DataFrame, str], None, None]: A generator of all possible drifts in the features and the description of the drift.
    """
    # Get features concept drift generators
    features_columns = [original_df[feature] for feature in drifting_features]
    generator_functions = [_get_feature_generator_function(column, feature_type) if feature_type else _get_feature_generator_function(column) 
                           for column, feature_type in zip(features_columns, drifting_features.values())]

    # Get the cartesian product of all drifts
    cartesian_products = lazy_product(generator_functions, args_lists=features_columns, args_type=SINGLE_ARGUMENT_EACH_GENERATOR)
    for drifts in cartesian_products:
        drifted_df = original_df.copy()
        drift_description = ""
        for drifted_column, current_description in drifts:
            drifted_df[drifted_column.name] = drifted_column
            drift_description += "_" + current_description
        yield (drifted_df, drift_description)