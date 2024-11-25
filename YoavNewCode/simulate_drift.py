import pandas as pd
import random

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

def _simulate_numeric_drift(
        df: pd.DataFrame, 
        feature: str
 ) -> list[tuple[pd.DataFrame, str]]:
    """
    Simulate all type of concept drifts in a numeric feature.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        feature (str): The column in which to insert the concept drift.

    Return:
        list[pd.DataFrame]: list of all generated concept drifts.
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
            pd.DataFrame: A new DataFrame with the concept drift applied.
        """
        return (df[feature] + k * feature_std, f"NumericFeature[{feature}{'+' if k >= 0 else '-'}{k}std]")
    
    
    #   Using it in iterations
    return [
        simulate_numeric_drift_of_size_k(k)
        for k in NUMERIC_DRIFT_SIZES
    ]



def _simulate_categorical_drift(
        df: pd.DataFrame, 
        feature: str
 ) -> list[tuple[pd.DataFrame, str]]:
    """
    Simulate concept drift in a specific categorical feature.
    The drift is simulated in every feature value in every relevant proportion size.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        feature (str): The column in which to insert concept drift.
    
    Returns:
        list[pd.DataFrame]: All types of drifts in the feature.
    """
    assert not pd.api.types.is_numeric_dtype(df[feature])
    
    unique_values = df[feature].unique()
    
    # Nested function
    def simulate_categorical_drift_in_value(
            fixed_value: str
     ) -> list[tuple[pd.DataFrame, str]]:
        """
        Simulate concept drift in a specific value of a feature (for all proportions).

        Parameters:
            fixed_value (str): The value that the drift is fixed to.

        Returns:
            list[pd.DataFrame]: a list of all drifts (in all proportions of the given value).
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
                pd.DataFrame: A new DataFrame with the concept drift applied.    
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
        return [
            simulate_categorical_drift_in_value_proportion(p)
            for p in CATEGORICAL_PROPORTIONS
        ]
    
    # Using the intermediate function
    drifted_dfs = []
    for feature_value in unique_values:
        drifted_dfs += simulate_categorical_drift_in_value(feature_value)
    return drifted_dfs

# Now the magic happens
def simulate_concept_drifts(
        original_df: pd.DataFrame, 
        drifting_features: list[str]
 ) -> list[pd.DataFrame]:
    """
    Simulate concept drift in a given list of features.
    Generate all types of drifts relevant to the features.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        drifting_features (list[str]): List of features to drift.

    Returns:
        list[pd.DataFrame]: List of all generated drifts.
    """
    result_drifts = [(original_df.copy(), "")]
    
    for feature in drifting_features:
        assert feature in original_df.columns

        # Get relevant drift function (by the feature type)
        drift_function = _simulate_numeric_drift if pd.api.types.is_numeric_dtype(original_df[feature]) else _simulate_categorical_drift
        
        if drift_function == _simulate_numeric_drift: # TODO - Fix error in numeric
            continue
        # Create new drift list
        generating_drifts = []
        for result_drift, result_description in result_drifts:
            print(result_description)
            for current_drift, current_description in drift_function(result_drift, feature):
                generating_drifts += [(current_drift, result_description + "_" + current_description)]
        result_drifts = generating_drifts
    
    return result_drifts

def save_all_possible_drifts(
        file_path, 
        path_prefix=""
 ) -> None:
    full_path = path_prefix + '\\' + file_path
    df = pd.read_csv(full_path)
    
    
    generated_drifts = simulate_concept_drifts(df, df.columns[: -1])  # TODO - Make possible of dynamically select features to drift

    file_prefix = file_path.split('.')[0]

    for generated_drift, drift_description in generated_drifts:
        new_path = path_prefix + "\\results\\" + file_prefix + drift_description + ".csv"
        generated_drift.to_csv(new_path)
    
if __name__ == "__main__":
    for file_path in FILE_PATHES:
        save_all_possible_drifts(file_path, DIRECTORY)