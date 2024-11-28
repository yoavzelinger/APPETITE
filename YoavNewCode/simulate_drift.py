import pandas as pd
import random
from typing import Generator

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
        df: pd.DataFrame, 
        feature: str
 ) -> Generator[tuple[pd.DataFrame, str], None, None]:
    """
    Generator for all type of concept drifts in a numeric feature.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        feature (str): The column in which to insert the concept drift.

    Return:
        Generator[tuple[pd.DataFrame, str], None, None]: A generator of all drifts in the feature and the description of the drift.
    """
    assert pd.api.types.is_numeric_dtype(df[feature])
    
    feature_std = df[feature].std()

    #   Nested function
    def simulate_numeric_drift_of_size_k(
            k: int
     ) -> tuple[pd.DataFrame, str]:
        """
        Simulate concept drift in a specific numeric feature of size k.

        Parameters:
            k (int): The value to add to all entries in the specified column.

        Returns:
            tuple[pd.DataFrame, str]: The new DataFrame with the concept drift and a description of the drift (in the format of "NumericFeature[feature{+,-}kstd]").
        """
        drifted_df = df.copy()
        drifted_df[feature] = drifted_df[feature] + k * feature_std
        return (drifted_df, f"NumericFeature[{feature}{'+' if k >= 0 else '-'}{k}std]")
    
    #   Using it in iterations
    for k in NUMERIC_DRIFT_SIZES:
        yield simulate_numeric_drift_of_size_k(k)



def _categorical_drift_generator(
        df: pd.DataFrame, 
        feature: str
 ) -> Generator[tuple[pd.DataFrame, str], None, None]:
    """
    Simulate concept drift in a specific categorical feature.
    The drift is simulated in every feature value in every relevant proportion size.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        feature (str): The column in which to insert concept drift.
    
    Returns:
        Generator[tuple[pd.DataFrame, str], None, None]: A generator of all drifts in the feature and the description of the drift.
    """
    assert not pd.api.types.is_numeric_dtype(df[feature])
    
    unique_values = df[feature].unique()
    
    # Nested function
    def categorical_drift_in_value_generator(
            fixed_value: str
     ) -> list[tuple[pd.DataFrame, str]]:
        """
        Simulate concept drift in a specific value of a feature (for all proportions).

        Parameters:
            fixed_value (str): The value that the drift is fixed to.

        Returns:
            list[tuple[pd.DataFrame, str]]: A list of all drifts in the feature (in all proportions of the given value) and the description of the drift.
        """
        assert fixed_value in unique_values

        #   Double nested function
        def simulate_categorical_drift_in_value_proportion(
                p: float
         ) -> tuple[pd.DataFrame, str]:
            """
            Simulate concept drift in a specific value of a feature for a given proportion.
            The Drift is done by fixing a given value for a proportion of the data.

            Parameters:
                p (float): the proportion of the samples (out of the remaining samples - do not contains the value) that the fixed value will be inserted.

            Returns:
                tuple[pd.DataFrame, str]: The new DataFrame with the concept drift and a description of the drift (in the format of "CategoricalFeature[feature=value;p=proportion]"). 
            """

            drifted_df = df.copy()
            remaining_indices = df[feature] != fixed_value
            remaining_count = remaining_indices.values.sum()
            remaining_indices = drifted_df[remaining_indices].index.values
            random.seed(10)
            fixed_indicies = random.choices(remaining_indices, k=int(remaining_count * p))
            drifted_df.loc[fixed_indicies, feature] = fixed_value

            return (drifted_df, f"CategoricalFeature[{feature}={fixed_value};p={str(p).replace('.',',')}]")
        
        #   Using the doubly nested function
        return (
            simulate_categorical_drift_in_value_proportion(p)
            for p in CATEGORICAL_PROPORTIONS
        )
    
    # Using the intermediate generator
    for feature_value in unique_values:
        for drifted_df in categorical_drift_in_value_generator(feature_value):
            yield drifted_df

# Now the magic happens
def simulate_concept_drifts(
        original_df: pd.DataFrame, 
        drifting_features: list[str]
        ) -> list[tuple[pd.DataFrame, str]]:
    """
    Simulate concept drift in a given list of features.
    Generate all types of drifts relevant to the features.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        drifting_features (list[str]): List of features to drift.

    Returns:
        list[tuple[pd.DataFrame, str]]: A list of all possible drifts in the features and the description of the drift.
    """
    result_drifts = [(original_df.copy(), "")]
    
    for feature in drifting_features:
        assert feature in original_df.columns

        # Get relevant drift function (by the feature type)
        drift_function = _numeric_drift_generator if pd.api.types.is_numeric_dtype(original_df[feature]) else _categorical_drift_generator
        
        # Create new drift list
        generating_drifts = []
        for result_drift, result_description in result_drifts:
            print(result_description)
            for current_drift, current_description in drift_function(result_drift, feature):
                generating_drifts += [(current_drift, result_description + "_" + current_description)]
        result_drifts = generating_drifts
    
    return result_drifts

def get_dataframe(file_path, 
        path_prefix=""
 ) -> pd.DataFrame:
    full_path = path_prefix + '\\' + file_path
    return pd.read_csv(full_path)
def save_all_possible_drifts(
        file_path, 
        path_prefix=""
 ) -> None:
    df = get_dataframe(file_path, path_prefix)
    
    generated_drifts = simulate_concept_drifts(df, df.columns[: -1])  # TODO - Make possible of dynamically select features to drift

    file_prefix = file_path.split('.')[0]

    for generated_drift, drift_description in generated_drifts:
        new_path = path_prefix + "\\results\\" + file_prefix + drift_description + ".csv"
        generated_drift.to_csv(new_path, index=False)
    
if __name__ == "__main__":
    for file_path in FILE_PATHES:
        save_all_possible_drifts(file_path, DIRECTORY)