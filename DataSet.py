import pandas as pd
from scipy.io import arff

class DataSet:

    def __init__(self, source_path, dataset_type, target_class, batch_size, feature_types, name=None, to_shuffle=False):
        # check data format
        if type(source_path) == str:
            self.name = source_path
            source_format = source_path.split(".")[-1]
            if source_format in ("csv", "data", "txt"):
                data_df = pd.read_csv(source_path)
            elif source_format == "arff":
                data, meta = arff.loadarff(source_path)
                pd_df = pd.DataFrame(data)
                pd_df[target_class] = pd_df[target_class].astype('int')
                data_df = pd_df
        else:  # already dataframe
            data_df = source_path
            self.name = name

        # convert categorical to nums
        for col in data_df:
            if data_df[col].dtype == object:
                data_df[col] = pd.Categorical(data_df[col])
                data_df[col] = data_df[col].cat.codes

        if to_shuffle:  # shuffle data, same shuffle always
            data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.data = data_df

        self.dataset_type = dataset_type
        self.features = list(self.data.columns)
        self.features.remove(target_class)
        self.target = target_class
        self.batch_size = batch_size
        self.feature_types = feature_types

if __name__ == '__main__':
    d = DataSet("data/sea.arff", None, "cl")
    print(d.features)
    d = DataSet("data/sea_0123_abrupto_noise_0.2.csv", None, "class")
    print(d.features)
