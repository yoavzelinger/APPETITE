import pandas as pd

class DataSet:

    def __init__(self, data_path, dataset_type, features, target):
        self.data = pd.read_csv(data_path)  # current data has header
        self.dataset_type = dataset_type
        self.features = features
        self.target = target
