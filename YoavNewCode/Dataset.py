import pandas as pd
from scipy.io import arff

PROPORTIONS_TUPLE = (0.7, 0.1, 0.2)
CONCEPT_PROPORTION, DRIFT_PROPOTION, TEST_PROPORTION = PROPORTIONS_TUPLE
RANDOM_STATE = 42

class Dataset:
    def __init__(self, 
                 source: str | pd.DataFrame, 
                 dataset_type: str = "", 
                 target_class: str = "", 
                 feature_types: list = None, 
                 size: int | tuple | list = PROPORTIONS_TUPLE, 
                 name: str = None, 
                 to_shuffle: bool = False
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
        feature_types: list
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
        feature_types = [] # Trying to fill from the data

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
            feature_types.append(column_type)
        
        if not target_class:
            # Taking last column as the target
            target_class = source.columns[-1]
        self.target = target_class

        if self.feature_types is None:
            # Removing the target column from the feature types
            target_index = source.columns.get_loc(target_class)
            feature_types.pop(target_index)
            self.feature_types = feature_types

        if to_shuffle:  # shuffle data, same shuffle always
            source = source.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        self.data = source

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

def get_ds() -> Dataset:
    directory = ""
    file_name = ""
    relative_path = directory + file_name
    return Dataset(relative_path)