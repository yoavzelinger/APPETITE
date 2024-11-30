import pandas as pd
from scipy.io import arff
from typing import Generator

from YoavNewCode.DataManagementTools.DriftSimulation import single_feature_concept_drift_generator, multiple_features_concept_drift_generator

PROPORTIONS_TUPLE = (0.7, 0.1, 0.2)
CONCEPT_PROPORTION, DRIFT_PROPOTION, TEST_PROPORTION = PROPORTIONS_TUPLE
RANDOM_STATE = 42

class Dataset:
    def __init__(self, 
                 source: str | pd.DataFrame, 
                 dataset_type: str = "", 
                 target_class: str = "", 
                 feature_types: dict[str, str] = None, 
                 size: int | tuple | list = PROPORTIONS_TUPLE, 
                 name: str = "", 
                 to_shuffle: bool = True
    ):
        """
        source: str or pd.DataFrame
            If str, the path to the dataset file
            If pd.DataFrame, the dataset itself
        dataset_type: str
            The type of the dataset
        target_class: str
            The target class column name
            If not provided the last column is used as the target
        feature_types: dict
            The types of the features
            If not provided then interpreted from the data
        size: int or tuple
            The size of the dataset
            If int, the size of the before concept
            If tuple, the size of the before concept, the size of the after concept, the size of the test concept
            If list, the size of the concept, the window size, the number of windows used, the test size and the slot size (Optional)
        name: str
            The name of the dataset
            If not provided then trying to interpret from the source (if it's a path)
        to_shuffle: bool
            Whether to shuffle the data
        """
        # Get data
        if type(source) == str:    # Path to the file
            file_name = source.split("\\")[-1]
            self.name, source_format = file_name.split(".")
            if source_format in ("csv", "data", "txt"):
                source = pd.read_csv(source)
            elif source_format == "arff":
                data, _ = arff.loadarff(source)
                source = pd.DataFrame(data)
        assert isinstance(source, pd.DataFrame)
        
        if name:
            self.name = name
        
        self.dataset_type = dataset_type
        
        self.feature_types = feature_types
        feature_types = {} # Trying to fill from the data

        for col in source:
            column_type = source[col].dtype
            if column_type in [object, bool]:   # Categorical or Binary
                source[col] = source[col].fillna(source[col].mode().iloc[0])    # Fill NaN values
                # Convert column to numeric
                if column_type == object:
                    source[col] = pd.Categorical(source[col])
                    source[col] = source[col].cat.codes
                    column_type ="categorical"
                else:
                    source[col] = source[col].replace({True: 1, False: 0})
                    column_type = "binary"
            else:   # Numeric
                source[col] = source[col].fillna(source[col].mean())    # Fill NaN values
                column_type = "numeric"
            feature_types[col] = column_type
        
        if not target_class:
            # Taking last column as the target
            target_class = source.columns[-1]
        self.target = target_class

        if self.feature_types is None:
            # Removing the target column from the feature types
            feature_types.pop(target_class)
            self.feature_types = feature_types

        if to_shuffle:  # shuffle data, same shuffle always
            source = source.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        self.data = source
        self.data.attrs["name"] = self.name

        n_samples = len(source)
        if type(size) == list:
            if len(size) == 4:
                concept_size, window, n_used, test = size
            elif len(size) == 5:
                concept_size, window, n_used, test, slot = size
                self.slot = slot
            self.before_size = concept_size - int(window/2)  # "clean" concept
            self.after_size = int(window*n_used)
            self.test_size = int(window*test)
            self.window = window
            self.concept_size = concept_size
        else:
            before_proprtion, after_proprtion, test_proprtion = CONCEPT_PROPORTION, DRIFT_PROPOTION, TEST_PROPORTION
            if type(size) == int:
                self.before_size = size
                before_proprtion = 1.0 * size / n_samples
                total_post_proportion = (1 - before_proprtion) / (after_proprtion + test_proprtion)
                after_proprtion, test_proprtion = after_proprtion * total_post_proportion, test_proprtion * total_post_proportion
            elif type(size) == tuple and len(size) == 3:
                before_proprtion, after_proprtion, test_proprtion = size
            self.before_size = int(before_proprtion*n_samples)
            self.after_size = int(after_proprtion*n_samples)
            self.test_size = int(test_proprtion*n_samples)

        assert (self.before_size + self.after_size + self.test_size) <= n_samples

    def split_features_targets(self, 
                               data: pd.DataFrame
     ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split the data to X and y
        
        Parameters:
            data (pd.DataFrame): The data to split
        
        Returns:
            tuple[pd.DataFrame, pd.Series]: The X and y
        """
        X = data.drop(columns=[self.target])
        y = data[self.target]
        return X, y

    def get_before_concept(self) -> tuple[pd.DataFrame, pd.Series]:
        before_concept_data = self.data.iloc[:self.before_size]
        return self.split_features_targets(before_concept_data)
    
    def get_after_concept(self) -> tuple[pd.DataFrame, pd.Series]:
        after_concept_data = self.data.iloc[self.before_size: self.before_size+self.after_size]
        return self.split_features_targets(after_concept_data)
    
    def get_test_concept(self) -> tuple[pd.DataFrame, pd.Series]:
        test_concept_data = self.data.iloc[-self.test_size:]
        return self.split_features_targets(test_concept_data)
    
    def drift_data_generator(self,
                   data: pd.DataFrame,
                   drift_features: str | list[str]
     ) -> Generator[pd.DataFrame, None, None]:
        """
        Create a drift in the data
        
        Parameters:
            data (pd.DataFrame): The data to drift
            drift_features (str or list): single feature or list of features to drift
                
        Returns:
            pd.DataFrame: The drifted data
        """
        if type(drift_features) == str:
            assert drift_features in data.columns, f"Feature {drift_features} not in the dataset"
            feature_type = self.feature_types[drift_features]
            return single_feature_concept_drift_generator(data, drift_features, feature_type)
        assert all([feature in data for feature in drift_features]), "Not all features in the dataset"
        # Get subset of the dictionary
        drift_features_dict = {feature: self.feature_types[feature] for feature in drift_features}
        return multiple_features_concept_drift_generator(data, drift_features_dict)

    def _drift_data_generator(self,
                   data: pd.DataFrame,
                   drift_features: str | list[str]
     ) -> Generator[pd.DataFrame, None, None]:
        """
        Create a drift in the data
        
        Parameters:
            data (pd.DataFrame): The data to drift
            drift_features (str or list): single feature or list of features to drift
                
        Returns:
            pd.DataFrame: The drifted data
        """
        if type(drift_features) == str:
            assert drift_features in data.columns, f"Feature {drift_features} not in the dataset"
            feature_type = self.feature_types[drift_features]
            return single_feature_concept_drift_generator(data, drift_features, feature_type)
        assert all([feature in data for feature in drift_features]), "Not all features in the dataset"
        # Get subset of the dictionary
        drift_features_dict = {feature: self.feature_types[feature] for feature in drift_features}
        return multiple_features_concept_drift_generator(data, drift_features_dict)
    
    partitions = ["before", "after", "test"]

    def partition_drift_generator(self,
                                  drift_features: str | list[str],
                                  partition: str = "after"
     ) -> Generator[tuple[pd.DataFrame, str], None, None]:
        """
        Drift generator for a specific partition
        
        Parameters:
            drift_features (str or list): single feature or list of features to drift
            partition (str): The partition to drift

        Returns:
            Generator[tuple[pd.DataFrame, str], None, None]: 
                A generator of all possible drifts in the feature and the description of the drift in the given partion name.
        """
        assert partition in self.partitions, "Invalid partition name"
        get_partion_dict = {
            "before": self.get_before_concept,
            "after": self.get_after_concept,
            "test": self.get_test_concept
            }
        original_X, y = get_partion_dict[partition]()
        for drifted_X, description in self._drift_data_generator(original_X, drift_features):
            yield (drifted_X, y), f"{partition.upper()}_{description}"