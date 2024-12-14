from pandas import read_csv, DataFrame, Series, Categorical, get_dummies
from scipy.io.arff import loadarff
from typing import Generator

from APPETITE.Constants import BEFORE_PROPORTION, AFTER_PROPORTION, TEST_PROPORTION, PROPORTIONS_TUPLE, RANDOM_STATE

from .DriftSimulation import single_feature_concept_drift_generator, multiple_features_concept_drift_generator

class Dataset:
    partitions = ["before", "after", "test"]

    def __init__(self, 
                 source: str | DataFrame,
                 size: int | tuple | list = PROPORTIONS_TUPLE,
                 to_shuffle: bool = True,
                 one_hot_encoding: bool = True
    ):
        """
        source: str or DataFrame
            If str, the path to the dataset file
            If DataFrame, the dataset itself
        size: int or tuple
            The size of the dataset
            If int, the size of the before concept
            If tuple, the size of the before concept, the size of the after concept, the size of the test concept
            If list, the size of the concept, the window size, the number of windows used, the test size and the slot size (Optional)
        to_shuffle: bool
            Whether to shuffle the data
        one_hot_encoding: bool
            Whether to use one hot encoding for the categorical
        """
        # Get data
        if type(source) == str:    # Path to the file
            file_name = source.split("\\")[-1]
            self.name, source_format = file_name.split(".")
            if source_format in ("csv", "data", "txt"):
                source = read_csv(source)
            elif source_format == "arff":
                data, _ = loadarff(source)
                source = DataFrame(data)
        assert isinstance(source, DataFrame)

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
            if column_type == bool or not one_hot_encoding:
                self.data[column_name] = Categorical(self.data[column_name])
                self.data[column_name] = self.data[column_name].cat.codes
                self.feature_types[column_name] = "binary" if column_type == bool else "categorical" 
                continue
            # One hot encoding with multiple values
            one_hot_encoded_dict.update({f"{column_name}_{value}": "binary" for value in self.data[column_name].unique()})
            self.data = get_dummies(self.data, columns=[column_name])
        self.feature_types.update(one_hot_encoded_dict)

        self.data[self.target_name] = Categorical(y.fillna(y.mode().iloc[0]))
        self.data[self.target_name] = self.data[self.target_name].cat.codes

        if to_shuffle:  # shuffle data - same shuffle always
            self.data = self.data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        self.data.attrs["name"] = self.name

        n_samples = len(self.data)
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
            before_proportion, after_proportion, test_proportion = BEFORE_PROPORTION, AFTER_PROPORTION, TEST_PROPORTION
            if type(size) == int:
                self.before_size = size
                before_proportion = 1.0 * size / n_samples
                total_post_proportion = (1 - before_proportion) / (after_proportion + test_proportion)
                after_proportion, test_proportion = after_proportion * total_post_proportion, test_proportion * total_post_proportion
            elif type(size) == tuple and len(size) == 3:
                before_proportion, after_proportion, test_proportion = size
            self.before_size = int(before_proportion*n_samples)
            self.after_size = int(after_proportion*n_samples)
            self.test_size = int(test_proportion*n_samples)

        assert (self.before_size + self.after_size + self.test_size) <= n_samples

    def split_features_targets(self, 
                               data: DataFrame
     ) -> tuple[DataFrame, Series]:
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

    def get_before_concept(self) -> tuple[DataFrame, Series]:
        before_concept_data = self.data.iloc[:self.before_size]
        return self.split_features_targets(before_concept_data)
    
    def get_after_concept(self) -> tuple[DataFrame, Series]:
        after_concept_data = self.data.iloc[self.before_size: self.before_size+self.after_size]
        return self.split_features_targets(after_concept_data)
    
    def get_test_concept(self) -> tuple[DataFrame, Series]:
        test_concept_data = self.data.iloc[-self.test_size:]
        return self.split_features_targets(test_concept_data)

    def _drift_data_generator(self,
                   data: DataFrame,
                   drift_features: str | list[str]
     ) -> Generator[DataFrame, None, None]:
        """
        Create a drift in the data
        
        Parameters:
            data (DataFrame): The data to drift
            drift_features (str or list): single feature or list of features to drift
                
        Returns:
            DataFrame: The drifted data
        """
        if type(drift_features) == str:
            assert drift_features in data.columns, f"Feature {drift_features} not in the dataset"
            feature_type = self.feature_types[drift_features]
            return single_feature_concept_drift_generator(data, drift_features, feature_type)
        assert all([feature in data for feature in drift_features]), "Not all features in the dataset"
        # Get subset of the dictionary
        drift_features_dict = {feature: self.feature_types[feature] for feature in drift_features}
        return multiple_features_concept_drift_generator(data, drift_features_dict)

    def drift_generator(self,
                                  drift_features: str | list[str],
                                  partition: str = "after"
     ) -> Generator[tuple[tuple[DataFrame, Series], str], None, None]:
        """
        Drift generator for a specific partition
        
        Parameters:
            drift_features (str or list): single feature or list of features to drift
            partition (str): The partition to drift

        Returns:
            Generator[tuple[tuple[DataFrame, Series], str], None, None]: 
                A generator of all possible drifts in the feature and the description of the drift in the given portion name.
                Each drift represented by the (drifted dataset, original y) and the description of the drift. 
        """
        assert partition in Dataset.partitions, "Invalid partition name"
        get_portion_dict = {
            "before": self.get_before_concept,
            "after": self.get_after_concept,
            "test": self.get_test_concept
        }
        original_X, y = get_portion_dict[partition]()
        for drifted_X, description in self._drift_data_generator(original_X, drift_features):
            yield (drifted_X, y), f"{partition.upper()}_{description}"

    def get_feature_first_drift(self,
                                  feature: str,
                                  partition: str = "after"
     ) -> Generator[tuple[tuple[DataFrame, Series], str], None, None]:
        drift_generator = self.drift_generator(feature, partition)
        return next(drift_generator)