import pandas as pd
import numpy as np

# source_path = "data/Classification_Datasets/visualizing_livestock.csv"
from sklearn import metrics
import os
import pickle

from DataSet import DataSet
from buildModel import build_model, prune_tree, map_tree

dataseds_dict = {}
sizes = (0.7, 0.1, 0.2)
directory = r"data\Classification_Datasets"

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]#.astype(np.float64)


for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        source_path = f
        # source_path = r"data\Classification_Datasets\analcatdata_boxing1.csv"
        dataseds_dict[source_path] = {}

        dataset_name = source_path.split("\\")[-1].split(".")[0]
        dataseds_dict[source_path]["name"] = dataset_name
        # print(f"---------{dataset_name}------------")

        data_df = pd.read_csv(source_path)
        feature_type = []
        target_name = None

        count_cat = 0
        count_num = 0
        for col in data_df:
            target_name = col
            if data_df[col].dtype == object:
                feature_type.append("categorical")
                count_cat += 1
                if pd.isna(data_df[col]).any():
                    data_df[col].fillna(data_df[col].mode().iloc[0], inplace=True)
            else:
                feature_type.append("numeric")
                count_num += 1
                if pd.isna(data_df[col]).any():
                    data_df[col].fillna((data_df[col].mean()), inplace=True)

        feature_type = feature_type[:-1]
        dataseds_dict[source_path]["number of features"] = len(feature_type)
        if data_df[target_name].dtype == object:
            count_cat -= 1
        else:
            count_num -= 1
        dataseds_dict[source_path]["categorical features"] = count_cat
        dataseds_dict[source_path]["numeric features"] = count_num

        # count class options
        options = data_df[target_name].unique()
        if len(options) > 2:
            dataseds_dict[source_path]["class type"] = "categorical"
        else:
            dataseds_dict[source_path]["class type"] = "binary"

        dataset = DataSet(data_df, "diagnosis_check", target_name, feature_type, sizes, name=dataset_name,
                          to_shuffle=True)
        pd.set_option('max_columns', None)

        dataseds_dict[source_path]["dataset size"] = len(data_df)

        concept_size = dataset.before_size
        train = dataset.data.iloc[0:int(0.9 * concept_size)]
        validation = dataset.data.iloc[int(0.9 * concept_size):concept_size]
        model = build_model(train, dataset.features, dataset.target, val_data=validation)

        val_x = validation[dataset.features]
        val_y = validation[dataset.target]
        prediction = model.predict(val_x)
        accuracy_val = metrics.accuracy_score(val_y, prediction)
        dataseds_dict[source_path]["accuracy validation"] = accuracy_val

        test = dataset.data.iloc[len(dataset.data) - dataset.test_size:-1]
        test_x = test[dataset.features]
        test_y = test[dataset.target]

        tree_rep = map_tree(model)
        dataseds_dict[source_path]["size before pruning"] = len(tree_rep)

        pruned_model = prune_tree(model, tree_rep)
        tree_rep = map_tree(model)
        dataseds_dict[source_path]["size after pruning"] = len(tree_rep)

        prediction = model.predict(test_x)
        accuracy_test = metrics.accuracy_score(test_y, prediction)
        dataseds_dict[source_path]["accuracy test"] = accuracy_test
        data_df = pd.read_csv(f)

print(dataseds_dict)

with open('dataseds_dict1.pickle', 'wb') as handle:
    pickle.dump(dataseds_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('dataseds_dict1.pickle', 'rb') as handle:
    d = pickle.load(handle)

assert dataseds_dict == d

