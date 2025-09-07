import os
from scipy.io.arff import loadarff

import pandas as pd
import math

from typing import Generator

import Tester.TesterConstants as tester_constants

from .DriftSimulation import single_feature_concept_drift_generator, multiple_features_concept_drift_generator

class Dataset:
    partitions = ["before", "after", "test"]

    def __init__(self, 
                 source: str | pd.DataFrame
    ):
        """
        source (str | DataFrame): If str, the path to the dataset file; If DataFrame, the dataset itself.
        """
        # Get data
        if type(source) == str:    # Path to the file
            self.name, source_format = os.path.splitext(os.path.basename(source))
            if source_format in (".csv", ".data", ".txt"):
                source = pd.read_csv(source)
            elif source_format == ".arff":
                data, _ = loadarff(source)
                source = pd.DataFrame(data)
        assert isinstance(source, pd.DataFrame)

        self.target_name = source.columns[-1]
        self.data, y = self.split_features_targets(source)

        self.feature_types = {}
        one_hot_encoded_dict = {}
        for column_name in self.data.columns:
            column_type = self.data[column_name].dtype
            if column_type not in [object, bool]:   # Numeric
                self.data[column_name] = self.data[column_name].fillna(self.data[column_name].mean())    # Fill NaN values
                self.feature_types[column_name] = "numeric"
                continue
            # Categorical or Binary
            self.data[column_name] = self.data[column_name].fillna(self.data[column_name].mode().iloc[0])    # Fill NaN values
            if len(self.data[column_name].unique()) <= 2: # Consider as binary
                column_type = bool
            if column_type == bool or not tester_constants.ONE_HOT_ENCODING_CATEGORICAL:
                self.data[column_name] = pd.Categorical(self.data[column_name])
                self.data[column_name] = self.data[column_name].cat.codes
                self.feature_types[column_name] = "binary" if column_type == bool else "categorical" 
                continue
            # One hot encoding with multiple values
            one_hot_encoded_dict.update({f"{column_name}_{value}": "binary" for value in self.data[column_name].unique()})
            self.data = pd.get_dummies(self.data, columns=[column_name])
        self.feature_types.update(one_hot_encoded_dict)

        self.data[self.target_name] = pd.Categorical(y.fillna(y.mode().iloc[0]))
        self.data[self.target_name] = self.data[self.target_name].cat.codes

            self.data = self.data.sample(frac=1, random_state=tester_constants.RANDOM_STATE).reset_index(drop=True)

        self.data.attrs["name"] = self.name

        n_samples = len(self.data)
        self.before_proportion, self.after_proportion, self.test_proportion = tester_constants.PROPORTIONS_TUPLE
        self.before_size = math.floor(self.before_proportion*n_samples)
        self.after_size = math.floor(self.after_proportion*n_samples)
        self.test_size = math.floor(self.test_proportion*n_samples)
        self.total_after_size = self.after_size + self.test_size

        assert all([0 < current_size for current_size in (self.before_size, self.after_size, self.test_size)])
        assert (self.before_size + self.after_size + self.test_size) <= n_samples

        self.update_after_window_size(tester_constants.DEFAULT_AFTER_WINDOW_PROPORTION)

    def update_after_window_size(self, 
                          new_after_window_size: int | float
     ) -> None:
        assert 0 < new_after_window_size and new_after_window_size <= 1

        self.after_window_proportion = new_after_window_size
        self.after_window_size = math.floor(self.after_size * self.after_window_proportion)

    def split_features_targets(self, 
                               data: pd.DataFrame
     ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split the data to X and y
        
        Parameters:
            data (DataFrame): The data to split
        
        Returns:
            tuple[DataFrame, Series]: The X and y
        """
        X = data.drop(columns=[self.target_name]).reset_index(drop=True)
        y = data[self.target_name].reset_index(drop=True)
        return X, y

    def get_before_concept(self) -> tuple[pd.DataFrame, pd.Series]:
        before_concept_data = self.data.iloc[:self.before_size]
        return self.split_features_targets(before_concept_data)
    
    def get_after_concept(self) -> tuple[pd.DataFrame, pd.Series]:
        after_concept_data = self.data.iloc[self.before_size: self.before_size + self.total_after_size]
        return self.split_features_targets(after_concept_data)
    
    def get_test_concept(self) -> tuple[pd.DataFrame, pd.Series]:
        test_concept_data = self.data.iloc[-self.test_size:]
        return self.split_features_targets(test_concept_data)
    
    def get_total_after_concept(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Get the total after concept data, including the after and test concepts
        
        Returns:
            tuple[DataFrame, Series]: The total after concept data
        """
        after_concept_data = self.data.iloc[self.before_size:]
        return self.split_features_targets(after_concept_data)

    def _drift_data_generator(self,
                   data: pd.DataFrame,
                   drift_features: str | list[str],
                   severity_levels: tuple = tester_constants.DEFAULT_GENERATED_SEVERITY_LEVELS
     ) -> Generator[pd.DataFrame, None, None]:
        """
        Create a drift in the data
        
        Parameters:
            data (DataFrame): The data to drift
            drift_features (str or list): single feature or list of features to drift
            severity_levels (tuple[int]): The severity levels the column should be drifted to. Default is all.
                
        Returns:
            DataFrame: The drifted data
        """
        if type(drift_features) == str:
            assert drift_features in data.columns, f"Feature {drift_features} not in the dataset"
            feature_type = self.feature_types[drift_features]
            return single_feature_concept_drift_generator(data, drift_features, feature_type, severity_levels)
        assert all([feature in data for feature in drift_features]), "Not all features in the dataset"
        # Get subset of the dictionary
        drift_features_dict = {feature: self.feature_types[feature] for feature in drift_features}
        return multiple_features_concept_drift_generator(data, drift_features_dict, severity_levels)

    def drift_generator(self,
                        drift_features: str | set[str],
                        partition: str = "after",
                        severity_levels: tuple = tester_constants.DEFAULT_GENERATED_SEVERITY_LEVELS
     ) -> Generator[tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], str, list[str]], None, None]:
        """
        Drift generator for a specific partition
        
        Parameters:
            drift_features (str or list): single feature or set of features to drift
            partition (str): The partition to drift.
            severity_levels (tuple[int]): The severity levels the column should be drifted to. Default is all.

        Returns:
            Generator[tuple[tuple[DataFrame, Series], str], None, None]: 
                A generator of all possible drifts in the feature and the description of the drift in the given partition name.
                Each drift represented by the (drifted dataset, original y) and the description of the drift and the drifted features.
        """
        assert partition in Dataset.partitions, "Invalid partition name"
        partition_function_mapping = {
            "before": self.get_before_concept,
            "after": self.get_total_after_concept
        }
        original_X, y = partition_function_mapping[partition]()
        for drifted_X, drift_severity_level, drift_description in self._drift_data_generator(original_X, drift_features, severity_levels):
            if partition == "before":
                yield (drifted_X, y), drift_severity_level, f"BEFORE_{drift_description}"
                continue
            # split to after and test
            X_after_drifted, y_after_drifted = drifted_X.iloc[:self.after_window_size], y.iloc[:self.after_window_size]
            X_test_drifted, y_test_drifted = drifted_X.iloc[self.after_size:], y.iloc[self.after_size:]
            yield (X_after_drifted, y_after_drifted), (X_test_drifted, y_test_drifted), drift_severity_level, drift_description

