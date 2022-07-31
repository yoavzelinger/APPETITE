import pandas as pd
import warnings
from datetime import datetime
from pandas.core.common import SettingWithCopyWarning
import numpy as np
from DataSet import DataSet
from buildModel import build_model, map_tree
from updateModel import print_tree_rules
from SingleTree import run_single_tree_experiment
from HiddenPrints import HiddenPrints
import xlsxwriter
import random

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
sizes = (0.7, 0.10, 0.2)
all_datasets = [
    DataSet("data/real/data_banknote_authentication.txt", "diagnosis_check", "class", ["numeric"] * 4, sizes, name="data_banknote_authentication", to_shuffle=True),
    DataSet("data/real/winequality-white.csv", "diagnosis_check", "quality", ["numeric"]*11, sizes, name="winequality-white", to_shuffle=True),
    DataSet("data/real/abalone.data", "diagnosis_check", "rings", ["categorical"] + ["numeric"]*7,  sizes, name="abalone", to_shuffle=True),
    #DataSet("data/real/iris.data", "diagnosis_check", "class", 100, ["numeric"]*4, name="iris", to_shuffle=True),
    DataSet("data/real/pima-indians-diabetes.csv", "diagnosis_check", "class", ["numeric"]*8, sizes, name="pima-indians-diabetes", to_shuffle=True)
]

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def manipulate_node(node, dataset, save_to_csv=False):
    feature_to_change_num = tree_rep[node]["feature"]
    type_of_feature = dataset.feature_types[feature_to_change_num]
    feature_to_change = dataset.features[int(feature_to_change_num)]
    print(f"changing feature: {feature_to_change} in node {node}")
    feature_in_path = False

    conditions = tree_rep[node]["condition"]
    verification_data = dataset.data.iloc[0:int(0.9 * concept_size)].copy()
    not_changed_data = dataset.data.iloc[0:concept_size].copy()
    filtered_data = dataset.data.iloc[concept_size:].copy()

    # filtering only node data
    indexes_filtered_data = (filtered_data[feature_to_change] > 0) | (filtered_data[feature_to_change] <= 0)
    indexes_verification_data = (verification_data[feature_to_change] > 0) | (verification_data[feature_to_change] <= 0)

    for cond in conditions:
        feature = cond["feature"]
        sign = cond["sign"]
        thresh = cond["thresh"]
        if feature == feature_to_change_num:
            feature_in_path = True
        feature_name = dataset.features[int(feature)]
        if sign == ">":
            indexes_filtered = filtered_data[feature_name] > thresh
            indexes_verification = verification_data[feature_name] > thresh
        else:  # <=
            indexes_filtered = filtered_data[feature_name] <= thresh
            indexes_verification = verification_data[feature_name] <= thresh
        indexes_filtered_data = indexes_filtered & indexes_filtered_data
        indexes_verification_data = indexes_verification & indexes_verification_data

    assert indexes_verification_data.sum() == model.tree_.n_node_samples[node], f"bad condition - node: {node}, cond: {conditions}" \
                                                                                f"\nnumber of samples should be {model.tree_.n_node_samples[node]} " \
                                                                                f"but it is {indexes_verification_data.sum()}"

    all_data = dataset.data
    # creating changes
    if type_of_feature == "numeric":
        # calculating statistics
        mean = float(all_data.mean()[feature_to_change])
        std = float(all_data.std()[feature_to_change])

        half_std_up = filtered_data.loc[indexes_filtered_data,feature_to_change] + 0.5*std
        half_std_down = filtered_data.loc[indexes_filtered_data,feature_to_change] - 0.5 * std
        one_std_up = filtered_data.loc[indexes_filtered_data,feature_to_change] + 1 * std
        one_std_down = filtered_data.loc[indexes_filtered_data,feature_to_change] - 1 * std
        two_std_up = filtered_data.loc[indexes_filtered_data, feature_to_change] + 2 * std
        two_std_down = filtered_data.loc[indexes_filtered_data, feature_to_change] - 2 * std
        feature_changes = [half_std_up, half_std_down, one_std_up, one_std_down, two_std_up, two_std_down]
        feature_changes_names = ["half_std_up", "half_std_down", "one_std_up", "one_std_down", "two_std_up", "two_std_down"]

    else:  # binary \ categorical
        values = all_data[feature_to_change].unique()
        value_counts = all_data[feature_to_change].value_counts()
        rows_to_change = indexes_filtered_data.sum()
        random.seed(17)
        uniform_dist = random.choices(values, weights=None, k=rows_to_change)

        distribution = np.zeros(len(values))
        for i in range(len(values)):
            val = values[i]
            distribution[i] = value_counts[val]
        distribution /= len(all_data)
        random.seed(5)
        orig_dist = random.choices(values, weights=distribution, k=rows_to_change)

        distribution3 = softmax(distribution)
        random.seed(31)
        softmax_orig_dist = random.choices(values, weights=distribution3, k=rows_to_change)

        values2 = filtered_data.loc[indexes_filtered_data,feature_to_change].unique()
        value_counts2 = filtered_data.loc[indexes_filtered_data,feature_to_change].value_counts()
        distribution2 = np.zeros(len(values2))
        for i in range(len(values2)):
            val = values2[i]
            distribution2[i] = value_counts2[val]
        distribution2 /= rows_to_change
        random.seed(13)
        filtered_dist = random.choices(values2, weights=distribution2, k=rows_to_change)

        distribution4 = softmax(distribution2)
        random.seed(7)
        softmax_filtered_dist = random.choices(values2, weights=distribution4, k=rows_to_change)

        feature_changes = [uniform_dist, orig_dist, filtered_dist, softmax_orig_dist, softmax_filtered_dist]
        feature_changes_names = ["uniform dist", "original dist", "filtered dist", "softmax orig dist", "softmax filtered dist"]

    # saving changes to csv \ yield
    for i in range(len(feature_changes)):
        change = feature_changes[i]
        change_name = feature_changes_names[i]
        to_save = filtered_data.copy()
        to_save.loc[indexes_filtered_data,feature_to_change] = change
        to_save = not_changed_data.append(to_save, ignore_index=True)
        yield to_save, change_name, type_of_feature, feature_in_path
        if save_to_csv:
            file_name = f'{dataset.name.split(".")[0]}_node_{node}_depth_{tree_rep[node]["depth"]}_{change_name}.csv'
            to_save.to_csv(file_name)

change_types = {
    "half_std_up": 0.5,
    "half_std_down": -0.5,
    "one_std_up": 1,
    "one_std_down": -1,
    "two_std_up": 2,
    "two_std_down": -2
}

severity = {
    "half_std_up": 1,
    "half_std_down": 1,
    "one_std_up": 2,
    "one_std_down": 2,
    "two_std_up": 3,
    "two_std_down": 3,
    "uniform dist": 2,
    "original dist": 2,
    "filtered dist": 1,
    "softmax orig dist": 2,
    "softmax filtered dist": 1
}

all_results = []
time_stamp = datetime.now()
date_time = time_stamp.strftime("%d-%m-%Y__%H-%M-%S")

for dataset in all_datasets:
    print(f"-------------{dataset.name.upper()}-------------")
    concept_size = dataset.before_size
    target = dataset.target
    feature_types = dataset.feature_types
    train = dataset.data.iloc[0:int(0.9 * concept_size)]
    validation = dataset.data.iloc[int(0.9 * concept_size):concept_size]
    model = build_model(train, dataset.features, dataset.target, val_data=validation)
    print("TREE:")
    print_tree_rules(model, dataset.features)

    node_list = list(range(model.tree_.node_count))
    print(f"tree size: {len(node_list)}")
    tree_rep = map_tree(model)
    non_leaf_nodes = list(filter(lambda n: tree_rep[n]["left"] != -1, node_list))
    print(non_leaf_nodes)
    leaf_nodes = list(filter(lambda n: tree_rep[n]["left"] == -1, node_list))
    print(leaf_nodes)

    # manipulate data & run experiment
    for node in non_leaf_nodes:
        print(f"node: {node}, depth: {tree_rep[node]['depth']}")
        manipulated_data = manipulate_node(node, dataset)

        for data, change, type_of_feature, feature_in_path in manipulated_data:
            dataset_for_exp = DataSet(data, "diagnosis_check", target, feature_types, sizes)
            with HiddenPrints():
                result = run_single_tree_experiment(dataset_for_exp, model=model, check_diagnosis=True, faulty_nodes=[node])
            result["dataset"] = dataset.name
            result["depth"] = tree_rep[node]['depth']
            result["samples in node"] = model.tree_.n_node_samples[node]
            result["change severity"] = severity[change]
            if change in change_types:
                change = change_types[change]
            result["change type"] = change
            result["feature type"] = type_of_feature
            result["number of faulty nodes"] = 1
            result["feature in path"] = feature_in_path
            all_results.append(result)

# write results to excel
file_name = f"results/result_run_{date_time}.xlsx"
workbook = xlsxwriter.Workbook(file_name)
worksheet = workbook.add_worksheet()
# write headers
dict_example = all_results[0]
index_col = {}
col_num = 0
for key in dict_example.keys():
    worksheet.write(0, col_num, key)
    index_col[key] = col_num
    col_num += 1
# write values
row_num = 1
for dict_res in all_results:
    for key, value in dict_res.items():
        if type(value) in (list, set, dict):
            value = str(value)
        col_num = index_col[key]
        try:
            worksheet.write(row_num, col_num, value)
        except TypeError:
            print(f"problem with key: '{key}', value: {value}")
    row_num += 1
workbook.close()

print("DONE")
