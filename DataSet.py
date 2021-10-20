import pandas as pd
from scipy.io import arff

class DataSet:

    def __init__(self, source_path, dataset_type, target_class):
        self.name = source_path

        # check data format
        source_format = source_path.split(".")[-1]
        print(source_format)
        if source_format == "csv":
            self.data = pd.read_csv(source_path)
        elif source_format == "arff":
            data, meta = arff.loadarff(source_path)
            self.data = pd.DataFrame(data)
        self.dataset_type = dataset_type
        self.features = list(self.data.columns)
        self.features.remove(target_class)
        self.target = target_class

if __name__ == '__main__':
    d = DataSet("data/sea.arff", None, "cl")
    print(d.features)
    d = DataSet("data/sea_0123_abrupto_noise_0.2.csv", None, "class")
    print(d.features)
