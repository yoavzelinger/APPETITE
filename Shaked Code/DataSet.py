import pandas as pd
from scipy.io import arff

class DataSet:

    def __init__(self, source_path, dataset_type, target_class, feature_types, size, name=None, to_shuffle=False):
        # check data format
        if type(source_path) == str:
            self.name = source_path
            source_format = source_path.split(".")[-1]
            if source_format in ("csv", "data", "txt"):
                data_df = pd.read_csv(source_path)
            elif source_format == "arff":
                data, meta = arff.loadarff(source_path)
                pd_df = pd.DataFrame(data)
                # pd_df[target_class] = pd_df[target_class].astype('int')
                data_df = pd_df
        else:  # already dataframe
            data_df = source_path
        if name is not None:
            self.name = name

        feature_type = []
        for col in data_df:
            target = col
            if data_df[col].dtype == object:
                feature_type.append("categorical")
                if pd.isna(data_df[col]).any():
                    data_df[col].fillna(data_df[col].mode().iloc[0], inplace=True)
                # convert categorical to nums
                data_df[col] = pd.Categorical(data_df[col])
                data_df[col] = data_df[col].cat.codes
            elif data_df[col].dtype == bool:
                data_df[col] = data_df[col].replace({True: 1, False: 0})
                feature_type.append("binary")
            else:
                feature_type.append("numeric")
                if pd.isna(data_df[col]).any():
                    data_df[col].fillna((data_df[col].mean()), inplace=True)

        if to_shuffle:  # shuffle data, same shuffle always
            data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.data = data_df

        if target_class is None:
            target_class = target
        self.dataset_type = dataset_type
        self.features = list(self.data.columns)
        self.features.remove(target_class)
        self.target = target_class

        if feature_types is None:
            feature_types = feature_type[:-1]
        self.feature_types = feature_types

        n_samples = len(data_df)

        if type(size) == tuple:
            before_size, after_size, test_size = size
            self.before_size = int(before_size*n_samples)
            self.after_size = int(after_size*n_samples)
            self.test_size = int(test_size*n_samples)

        elif type(size) == int:
            self.before_size = size
            self.after_size = int(size * 0.1)
            self.test_size = int(size * 0.2)

        elif type(size) == list:
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

        assert (self.before_size + self.after_size + self.test_size) <= n_samples

if __name__ == '__main__':
    # all_sizes = [
    #     (0.4, 0.4, 0.2),
    #     # (0.7, 0.1, 0.2),
    #     # (0.7, 0.07, 0.2),
    #     # (0.7, 0.05, 0.2),
    #     # (0.7, 0.02, 0.2)
    # ]
    # for sizes in all_sizes:
    #     all_datasets = [
    #         DataSet("data/real/iris.data", "diagnosis_check", "class", ["numeric"] * 4, sizes, name="iris",
    #                 to_shuffle=True),
    #         DataSet("data/real/data_banknote_authentication.txt", "diagnosis_check", "class", ["numeric"] * 4, sizes,
    #                 name="data_banknote_authentication", to_shuffle=True),
    #          DataSet("data/real/pima-indians-diabetes.csv", "diagnosis_check", "class", ["numeric"] * 8, sizes,
    #                 name="pima-indians-diabetes", to_shuffle=True)
    #     ]
    #
    #     for dataset in all_datasets:
    #         print(f"--------{dataset.name} {sizes}----------")
    #         data = dataset.data
    #         print("all dataset")
    #         print(f" values: {data[dataset.target].unique()} count:\n{data[dataset.target].value_counts()}")
    #
    #         before = dataset.data.iloc[:dataset.before_size]
    #         print("before")
    #         print(f" values: {before[dataset.target].unique()} count: \n{before[dataset.target].value_counts()}")
    #
    #         after = dataset.data.iloc[dataset.before_size: dataset.before_size + dataset.after_size]
    #         print("after")
    #         print(f" values: {after[dataset.target].unique()} count: \n{after[dataset.target].value_counts()}")
    #
    #         test = dataset.data.iloc[len(dataset.data) - dataset.test_size:-1]
    #         print("test")
    #         print(f" values: {test[dataset.target].unique()} count: \n{test[dataset.target].value_counts()}")

    DataSet("data/Classification_Datasets/kc3.csv", "diagnosis_check", "c", None, (0.7,0.1,0.2), name="kc3", to_shuffle=True)