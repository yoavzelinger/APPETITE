import pandas as pd
import warnings
from datetime import datetime
from pandas.core.common import SettingWithCopyWarning
import numpy as np
from DataSet import DataSet
from buildModel import build_model
from updateModel import print_tree_rules
from SingleTree import run_single_tree_experiment
from HiddenPrints import HiddenPrints
import xlsxwriter


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

source_path = "data/real/winequality-white.csv"
data = pd.read_csv(source_path)
print(list(data.columns[:-1]))
print(type(data.columns[-1]))

target = data.columns[-1]
features = list(data.columns[:-1])
feature_types = ["numeric"]*len(features)
concept_size = 2000
dataset = DataSet(source_path, "diagnosis_check", target, concept_size, feature_types)

model = build_model(dataset.data.iloc[0:concept_size], dataset.features, dataset.target, to_split=False)

print("TREE:")
print_tree_rules(model, dataset.features)


def map_tree(tree):
    tree_representation = {0: {"depth": 0,
                "parent": -1}}
    nodes_to_check = [0]
    while len(nodes_to_check) > 0:
        node = nodes_to_check.pop(0)

        left_child = tree.tree_.children_left[node]
        tree_representation[node]["left"] = left_child
        if left_child != -1:
            tree_representation[left_child] = {"parent":node,
                                               "type": "left"}
            nodes_to_check.append(left_child)
        right_child = tree.tree_.children_right[node]
        tree_representation[node]["right"] = right_child
        if right_child != -1:
            tree_representation[right_child] = {"parent":node,
                                               "type": "right"}
            nodes_to_check.append(right_child)

        tree_representation[node]["feature"] = model.tree_.feature[node]
        tree_representation[node]["threshold"] = model.tree_.threshold[node]

        if node != 0:
            parent = tree_representation[node]["parent"]
            tree_representation[node]["depth"] = tree_representation[parent]["depth"] + 1
            parent_cond = tree_representation[parent]["condition"]
            sign = "<=" if tree_representation[node]["type"] == "left" else ">"
            cond = f"{model.tree_.feature[parent]} {sign} {model.tree_.threshold[parent]}"
            tree_representation[node]["condition"] = parent_cond + [cond]
        else: #root
            tree_representation[node]["condition"] = []

    return tree_representation


node_list = list(range(model.tree_.node_count))
tree_rep = map_tree(model)
non_leaf_nodes = list(filter(lambda n: tree_rep[n]["left"] != -1, node_list))
print(non_leaf_nodes)
leaf_nodes = list(filter(lambda n: tree_rep[n]["left"] == -1, node_list))
print(leaf_nodes)

def manipulate_node(node, dataset):
    feature_to_change = tree_rep[node]["feature"]
    type_of_feature = dataset.feature_types[feature_to_change]
    feature_to_change = dataset.features[int(feature_to_change)]
    print(f"changing feature: {feature_to_change} in node {node}")

    conditions = tree_rep[node]["condition"]
    verification_data = dataset.data.iloc[0:concept_size].copy()
    filtered_data = dataset.data.iloc[concept_size:].copy()

    # filtering only node data
    indexes_filtered_data = (filtered_data[feature_to_change] > 0) | (filtered_data[feature_to_change] <= 0)
    indexes_verification_data = (verification_data[feature_to_change] > 0) | (verification_data[feature_to_change] <= 0)
    for cond in conditions:
        feature, sign, thresh = cond.split()
        thresh = np.float32(thresh)
        feature_name = dataset.features[int(feature)]
        if sign == ">":
            indexes_filtered = filtered_data[feature_name] > thresh
            indexes_verification = verification_data[feature_name] > thresh
            # filtered_data = filtered_data[filtered_data[feature_name] > thresh]
            # verification_data = verification_data[verification_data[feature_name] > thresh]
        else:  # <=
            indexes_filtered = filtered_data[feature_name] <= thresh
            indexes_verification = verification_data[feature_name] <= thresh
            # filtered_data = filtered_data[filtered_data[feature_name] <= thresh]
            # verification_data = verification_data[verification_data[feature_name] <= thresh]
        indexes_filtered_data = indexes_filtered & indexes_filtered_data
        indexes_verification_data = indexes_verification & indexes_verification_data
    # assert len(verification_data) == model.tree_.n_node_samples[node], f"bad condition - node: {node}, cond: {conditions}"
    assert indexes_verification_data.sum() == model.tree_.n_node_samples[node], f"bad condition - node: {node}, cond: {conditions}" \
                                                                                f"\nnumber of samples should be {model.tree_.n_node_samples[node]} but it is {indexes_verification_data.sum()}"

    # calculating statistics
    all_data = dataset.data
    mean = float(all_data.mean()[feature_to_change])
    std = float(all_data.std()[feature_to_change])

    # creating changes
    half_std_up = filtered_data.loc[indexes_filtered_data,feature_to_change] + 0.5*std
    half_std_down = filtered_data.loc[indexes_filtered_data,feature_to_change] - 0.5 * std
    one_std_up = filtered_data.loc[indexes_filtered_data,feature_to_change] + 1 * std
    one_std_down = filtered_data.loc[indexes_filtered_data,feature_to_change] - 1 * std
    feature_changes = [half_std_up, half_std_down, one_std_up, one_std_down]
    feature_changes_names = ["half_std_up", "half_std_down", "one_std_up", "one_std_down"]


    # saving changes to csv
    for i in range(len(feature_changes)):
        change = feature_changes[i]
        change_name = feature_changes_names[i]
        to_save = filtered_data.copy()
        to_save.loc[indexes_filtered_data,feature_to_change] = change
        to_save = verification_data.append(to_save, ignore_index=True)
        yield to_save, change_name, type_of_feature
        # file_name = f'{dataset.name.split(".")[0]}_node_{node}_depth_{tree_rep[node]["depth"]}_{change_name}.csv'
        # to_save.to_csv(file_name)

    #return filtered_data

change_types = {
    "half_std_up": 0.5,
    "half_std_down": -0.5,
    "one_std_up": 1,
    "one_std_down": -1
}
all_results = []
time_stamp = datetime.now()
date_time = time_stamp.strftime("%d-%m-%Y__%H-%M-%S")


for node in non_leaf_nodes:
    print(f"node: {node}, depth: {tree_rep[node]['depth']}")
    manipulated_data = manipulate_node(node, dataset)

    for data, change, type_of_feature in manipulated_data:
        dataset = DataSet(data, "diagnosis_check", target, concept_size, feature_types)
        with HiddenPrints():
            result = run_single_tree_experiment(dataset, model=model, check_diagnosis=True, faulty_nodes=[node])
        result["depth"] = tree_rep[node]['depth']
        result["samples in node"] = model.tree_.n_node_samples[node]
        result["change type"] = change_types[change]
        result["feature type"] = type_of_feature
        result["number of faulty nodes"] = 1
        all_results.append(result)
        
print(date_time)
# results_df = pd.DataFrame(all_results)
# file_name = f"results/result_run_{date_time}.csv"
# results_df.to_csv(file_name)

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
        if type(value) == list:
            value = str(value)
        col_num = index_col[key]
        worksheet.write(row_num, col_num, value)
    row_num += 1
workbook.close()

print("DONE")
