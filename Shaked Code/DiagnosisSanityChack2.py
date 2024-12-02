import sys

import pandas as pd
import warnings
from datetime import datetime
from pandas.core.common import SettingWithCopyWarning
import numpy as np
from sklearn import metrics

from DataSet import DataSet
from buildModel import build_model, map_tree, prune_tree
from updateModel import print_tree_rules
from SingleTree import run_single_tree_experiment
from HiddenPrints import HiddenPrints
import xlsxwriter
import random
import copy

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
epsilon = np.finfo(np.float64).eps

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
    max_value = None
    min_value = None

    concept_size = dataset.before_size
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
            if sign == ">":
                min_value = thresh
            else:
                max_value = thresh
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
        # assuring the drift will not affect other nodes
        if min_value is not None:
            if type_of_feature != "numeric":
                c1 = int(min_value + 0.5)
                change = [c1 if c < min_value else c for c in change]
                assert len(list(filter(lambda x: x < min_value, change))) == 0
            else:
                change[change < min_value] = min_value
                assert (change >= min_value).all(), f"node: {node}, conditions: {conditions}\nrangs: {min_value}-{max_value}\n{change}"
        if max_value is not None:
            if type_of_feature != "numeric":
                c1 = int(max_value - 0.5)
                change = [c1 if c >= max_value else c for c in change]
                assert len(list(filter(lambda x: x >= max_value, change))) == 0
            else:
                change[change >= max_value] = max_value - 0.0001
                assert (change < max_value).all(), f"node: {node}, conditions: {conditions}\nrangs: {min_value}-{max_value}\n{change}"

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


if __name__ == '__main__':
    time_stamp = datetime.now()
    date_time = time_stamp.strftime("%d-%m-%Y__%H-%M-%S")

    #run_name = sys.argv[1]
    run_name = "jaccard"

    file_name = f"results/{run_name}_result_run_{date_time}.xlsx"
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    empty_file = True
    row_num = 1

    all_sizes = [
        (0.7, 0.1, 0.2),
        (0.7, 0.07, 0.2),
        (0.7, 0.05, 0.2),
        (0.7, 0.02, 0.2)
    ]

    for sizes in all_sizes:
        # all_datasets = [
        #     DataSet("data/real/iris.data", "diagnosis_check", "class", ["numeric"] * 4, sizes, name="iris", to_shuffle=True),
        #     #DataSet("data/real/winequality-white.csv", "diagnosis_check", "quality", ["numeric"] * 11, sizes, name="winequality-white", to_shuffle=True),
        #     DataSet("data/real/data_banknote_authentication.txt", "diagnosis_check", "class", ["numeric"] * 4, sizes, name="data_banknote_authentication", to_shuffle=True),
        #     #DataSet("data/real/abalone.data", "diagnosis_check", "rings", ["categorical"] + ["numeric"]*7,  sizes, name="abalone", to_shuffle=True),
        #     DataSet("data/real/pima-indians-diabetes.csv", "diagnosis_check", "class", ["numeric"]*8, sizes, name="pima-indians-diabetes", to_shuffle=True)
        # ]

        all_datasets = pd.read_csv('data/all_datasets.csv', index_col=0)

        for index, row in all_datasets.iterrows():
            if index > 1:
                break
            dataset = DataSet(row["path"].replace("\\", "/"), "diagnosis_check", None, None, sizes, name=row["name"],
                          to_shuffle=True)

            all_results = []

        # for dataset in all_datasets:
            try:
                print(f"-------------{dataset.name.upper()} {sizes}-------------")
                concept_size = dataset.before_size
                target = dataset.target
                feature_types = dataset.feature_types
                train = dataset.data.iloc[0:int(0.9 * concept_size)].copy()
                validation = dataset.data.iloc[int(0.9 * concept_size):concept_size].copy()
                model = build_model(train, dataset.features, dataset.target, val_data=validation)
                tree_rep = map_tree(model)
                model = prune_tree(model, tree_rep)
                # print("TREE:")
                # print_tree_rules(model, dataset.features)

                after_samples = dataset.data.iloc[concept_size:concept_size + dataset.after_size].copy()
                after_data_x = after_samples[dataset.features]
                prediction = model.predict(after_data_x)
                after_data_y = after_samples[dataset.target]
                accuracy_after_no_drift = metrics.accuracy_score(after_data_y, prediction)

                test_samples = dataset.data.iloc[len(dataset.data) - dataset.test_size: -1].copy()
                test_data_x = test_samples[dataset.features]
                prediction = model.predict(test_data_x)
                test_data_y = test_samples[dataset.target]
                accuracy_test_no_drift = metrics.accuracy_score(test_data_y, prediction)

                tree_rep = map_tree(model)
                node_list = list(tree_rep.keys())
                # print(f"tree size: {len(node_list)}")
                non_leaf_nodes = list(filter(lambda n: tree_rep[n]["left"] != -1, node_list))
                # print(non_leaf_nodes)
                leaf_nodes = list(filter(lambda n: tree_rep[n]["left"] == -1, node_list))
                # print(leaf_nodes)

                data_before_manipulation = dataset.data.iloc[0:concept_size]
                # manipulate data & run experiment
                for node in non_leaf_nodes:
                    # print(f"node: {node}, depth: {tree_rep[node]['depth']}")
                    manipulated_data = manipulate_node(node, dataset)

                    for data, change, type_of_feature, feature_in_path in manipulated_data:
                        dataset_for_exp = DataSet(data, "diagnosis_check", target, feature_types, sizes)
                        data_after_manipulation = dataset_for_exp.data.iloc[0:concept_size]
                        assert data_after_manipulation.equals(data_before_manipulation.astype(data_after_manipulation.dtypes)), \
                            f"before:\n{data_before_manipulation}\n\nafter:\n{data_after_manipulation}"

                        with HiddenPrints():
                            result = run_single_tree_experiment(dataset_for_exp, model=copy.deepcopy(model), check_diagnosis=True, faulty_nodes=[node], name=run_name)
                        result["size"] = sizes[1]
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
                        result["model accuracy - no drift - after"] = accuracy_after_no_drift
                        result["model accuracy - no drift - test"] = accuracy_test_no_drift
                        all_results.append(result)
            except Exception as e:
                print(f"failed in run: {dataset.name.upper()} {sizes}, change type: {change}, node:{node}")
                print(e)

            # write results to excel
            if empty_file:
                # write headers
                dict_example = all_results[0]
                index_col = {}
                col_num = 0
                for key in dict_example.keys():
                    worksheet.write(0, col_num, key)
                    index_col[key] = col_num
                    col_num += 1
                empty_file = False
            # write values after every dataset
            for dict_res in all_results:
                for key, value in dict_res.items():
                    if type(value) in (list, set, dict):
                        value = str(value)
                    col_num = index_col[key]
                    try:
                        worksheet.write(row_num, col_num, value)
                    except TypeError:
                        print(f"{dataset.name.upper()} {sizes} - problem with key: '{key}', value: {value}")
                row_num += 1
    workbook.close()
    print("DONE")


# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
# epsilon = np.finfo(np.float64).eps
#
#
# def softmax(x):
#     y = np.exp(x - np.max(x))
#     f_x = y / np.sum(np.exp(x))
#     return f_x
#
#
# def manipulate_node(node, dataset, save_to_csv=False):
#     feature_to_change_num = tree_rep[node]["feature"]
#     type_of_feature = dataset.feature_types[feature_to_change_num]
#     feature_to_change = dataset.features[int(feature_to_change_num)]
#     print(f"changing feature: {feature_to_change} in node {node}")
#     feature_in_path = False
#     max_value = None
#     min_value = None
#
#     concept_size = dataset.before_size
#     conditions = tree_rep[node]["condition"]
#     verification_data = dataset.data.iloc[0:int(0.9 * concept_size)].copy()
#     not_changed_data = dataset.data.iloc[0:concept_size].copy()
#     filtered_data = dataset.data.iloc[concept_size:].copy()
#
#     # filtering only node data
#     indexes_filtered_data = (filtered_data[feature_to_change] > 0) | (filtered_data[feature_to_change] <= 0)
#     indexes_verification_data = (verification_data[feature_to_change] > 0) | (verification_data[feature_to_change] <= 0)
#
#     for cond in conditions:
#         feature = cond["feature"]
#         sign = cond["sign"]
#         thresh = cond["thresh"]
#         if feature == feature_to_change_num:
#             feature_in_path = True
#             if sign == ">":
#                 min_value = thresh
#             else:
#                 max_value = thresh
#         feature_name = dataset.features[int(feature)]
#         if sign == ">":
#             indexes_filtered = filtered_data[feature_name] > thresh
#             indexes_verification = verification_data[feature_name] > thresh
#         else:  # <=
#             indexes_filtered = filtered_data[feature_name] <= thresh
#             indexes_verification = verification_data[feature_name] <= thresh
#         indexes_filtered_data = indexes_filtered & indexes_filtered_data
#         indexes_verification_data = indexes_verification & indexes_verification_data
#
#     assert indexes_verification_data.sum() == model.tree_.n_node_samples[
#         node], f"bad condition - node: {node}, cond: {conditions}" \
#                f"\nnumber of samples should be {model.tree_.n_node_samples[node]} " \
#                f"but it is {indexes_verification_data.sum()}"
#
#     all_data = dataset.data
#     # creating changes
#     if type_of_feature == "numeric":
#         # calculating statistics
#         mean = float(all_data.mean()[feature_to_change])
#         std = float(all_data.std()[feature_to_change])
#
#         half_std_up = filtered_data.loc[indexes_filtered_data, feature_to_change] + 0.5 * std
#         half_std_down = filtered_data.loc[indexes_filtered_data, feature_to_change] - 0.5 * std
#         one_std_up = filtered_data.loc[indexes_filtered_data, feature_to_change] + 1 * std
#         one_std_down = filtered_data.loc[indexes_filtered_data, feature_to_change] - 1 * std
#         two_std_up = filtered_data.loc[indexes_filtered_data, feature_to_change] + 2 * std
#         two_std_down = filtered_data.loc[indexes_filtered_data, feature_to_change] - 2 * std
#         feature_changes = [half_std_up, half_std_down, one_std_up, one_std_down, two_std_up, two_std_down]
#         feature_changes_names = ["half_std_up", "half_std_down", "one_std_up", "one_std_down", "two_std_up",
#                                  "two_std_down"]
#
#     else:  # binary \ categorical
#         values = all_data[feature_to_change].unique()
#         value_counts = all_data[feature_to_change].value_counts()
#         rows_to_change = indexes_filtered_data.sum()
#         random.seed(17)
#         uniform_dist = random.choices(values, weights=None, k=rows_to_change)
#
#         distribution = np.zeros(len(values))
#         for i in range(len(values)):
#             val = values[i]
#             distribution[i] = value_counts[val]
#         distribution /= len(all_data)
#         random.seed(5)
#         orig_dist = random.choices(values, weights=distribution, k=rows_to_change)
#
#         distribution3 = softmax(distribution)
#         random.seed(31)
#         softmax_orig_dist = random.choices(values, weights=distribution3, k=rows_to_change)
#
#         values2 = filtered_data.loc[indexes_filtered_data, feature_to_change].unique()
#         value_counts2 = filtered_data.loc[indexes_filtered_data, feature_to_change].value_counts()
#         distribution2 = np.zeros(len(values2))
#         for i in range(len(values2)):
#             val = values2[i]
#             distribution2[i] = value_counts2[val]
#         distribution2 /= rows_to_change
#         random.seed(13)
#         filtered_dist = random.choices(values2, weights=distribution2, k=rows_to_change)
#
#         distribution4 = softmax(distribution2)
#         random.seed(7)
#         softmax_filtered_dist = random.choices(values2, weights=distribution4, k=rows_to_change)
#
#         feature_changes = [uniform_dist, orig_dist, filtered_dist, softmax_orig_dist, softmax_filtered_dist]
#         feature_changes_names = ["uniform dist", "original dist", "filtered dist", "softmax orig dist",
#                                  "softmax filtered dist"]
#
#     # saving changes to csv \ yield
#     for i in range(len(feature_changes)):
#         change = feature_changes[i]
#         # assuring the drift will not affect other nodes
#         if min_value is not None:
#             if type_of_feature != "numeric":
#                 c1 = int(min_value - 0.5)
#                 change = [c1 if c < min_value else c for c in change]
#                 assert len(list(filter(lambda x: x < min_value, change))) == 0
#             else:
#                 change[change < min_value] = min_value
#                 assert (
#                             change >= min_value).all(), f"node: {node}, conditions: {conditions}\nrangs: {min_value}-{max_value}\n{change}"
#         if max_value is not None:
#             if type_of_feature != "numeric":
#                 c1 = int(max_value - 0.5)
#                 change = [c1 if c >= max_value else c for c in change]
#                 assert len(list(filter(lambda x: x >= max_value, change))) == 0
#             else:
#                 change[change >= max_value] = max_value - 0.0001
#                 assert (
#                             change < max_value).all(), f"node: {node}, conditions: {conditions}\nrangs: {min_value}-{max_value}\n{change}"
#
#         change_name = feature_changes_names[i]
#         to_save = filtered_data.copy()
#         to_save.loc[indexes_filtered_data, feature_to_change] = change
#         to_save = not_changed_data.append(to_save, ignore_index=True)
#         yield to_save, change_name, type_of_feature, feature_in_path
#         if save_to_csv:
#             file_name = f'{dataset.name.split(".")[0]}_node_{node}_depth_{tree_rep[node]["depth"]}_{change_name}.csv'
#             to_save.to_csv(file_name)
#
#
# change_types = {
#     "half_std_up": 0.5,
#     "half_std_down": -0.5,
#     "one_std_up": 1,
#     "one_std_down": -1,
#     "two_std_up": 2,
#     "two_std_down": -2
# }
#
# severity = {
#     "half_std_up": 1,
#     "half_std_down": 1,
#     "one_std_up": 2,
#     "one_std_down": 2,
#     "two_std_up": 3,
#     "two_std_down": 3,
#     "uniform dist": 2,
#     "original dist": 2,
#     "filtered dist": 1,
#     "softmax orig dist": 2,
#     "softmax filtered dist": 1
# }
#
# if __name__ == '__main__':
#     all_results = []
#     time_stamp = datetime.now()
#     date_time = time_stamp.strftime("%d-%m-%Y__%H-%M-%S")
#
#     all_sizes = [
#         (0.7, 0.1, 0.2),
#         (0.7, 0.07, 0.2),
#         (0.7, 0.05, 0.2),
#         (0.7, 0.02, 0.2)
#     ]
#
#     for sizes in all_sizes:
#         # all_datasets = [
#         #     DataSet("data/real/iris.data", "diagnosis_check", "class", ["numeric"] * 4, sizes, name="iris", to_shuffle=True),
#         #     #DataSet("data/real/winequality-white.csv", "diagnosis_check", "quality", ["numeric"] * 11, sizes, name="winequality-white", to_shuffle=True),
#         #     DataSet("data/real/data_banknote_authentication.txt", "diagnosis_check", "class", ["numeric"] * 4, sizes, name="data_banknote_authentication", to_shuffle=True),
#         #     #DataSet("data/real/abalone.data", "diagnosis_check", "rings", ["categorical"] + ["numeric"]*7,  sizes, name="abalone", to_shuffle=True),
#         #     DataSet("data/real/pima-indians-diabetes.csv", "diagnosis_check", "class", ["numeric"]*8, sizes, name="pima-indians-diabetes", to_shuffle=True)
#         # ]
#
#         all_datasets = pd.read_csv('data/all_datasets.csv', index_col=0)
#
#         for index, row in all_datasets.iterrows():
#             # if index > 1:
#             #     break
#             dataset = DataSet(row["path"].replace("\\", "/"), "diagnosis_check", None, None, sizes, name=row["name"],
#                               to_shuffle=True)
#
#             # for dataset in all_datasets:
#
#             print(f"-------------{dataset.name.upper()} {sizes}-------------")
#             concept_size = dataset.before_size
#             target = dataset.target
#             feature_types = dataset.feature_types
#             train = dataset.data.iloc[0:int(0.9 * concept_size)].copy()
#             validation = dataset.data.iloc[int(0.9 * concept_size):concept_size].copy()
#             model = build_model(train, dataset.features, dataset.target, val_data=validation)
#             tree_rep = map_tree(model)
#             model = prune_tree(model, tree_rep)
#             print("TREE:")
#             print_tree_rules(model, dataset.features)
#
#             after_samples = dataset.data.iloc[concept_size:concept_size + dataset.after_size].copy()
#             after_data_x = after_samples[dataset.features]
#             prediction = model.predict(after_data_x)
#             after_data_y = after_samples[dataset.target]
#             accuracy_after_no_drift = metrics.accuracy_score(after_data_y, prediction)
#
#             test_samples = dataset.data.iloc[len(dataset.data) - dataset.test_size: -1].copy()
#             test_data_x = test_samples[dataset.features]
#             prediction = model.predict(test_data_x)
#             test_data_y = test_samples[dataset.target]
#             accuracy_test_no_drift = metrics.accuracy_score(test_data_y, prediction)
#
#             tree_rep = map_tree(model)
#             node_list = list(tree_rep.keys())
#             print(f"tree size: {len(node_list)}")
#             non_leaf_nodes = list(filter(lambda n: tree_rep[n]["left"] != -1, node_list))
#             print(non_leaf_nodes)
#             leaf_nodes = list(filter(lambda n: tree_rep[n]["left"] == -1, node_list))
#             print(leaf_nodes)
#
#             data_before_manipulation = dataset.data.iloc[0:concept_size]
#             # manipulate data & run experiment
#             for node in non_leaf_nodes:
#                 print(f"node: {node}, depth: {tree_rep[node]['depth']}")
#                 manipulated_data = manipulate_node(node, dataset)
#
#                 for data, change, type_of_feature, feature_in_path in manipulated_data:
#                     dataset_for_exp = DataSet(data, "diagnosis_check", target, feature_types, sizes)
#                     data_after_manipulation = dataset_for_exp.data.iloc[0:concept_size]
#                     assert data_after_manipulation.equals(
#                         data_before_manipulation.astype(data_after_manipulation.dtypes)), \
#                         f"before:\n{data_before_manipulation}\n\nafter:\n{data_after_manipulation}"
#
#                     with HiddenPrints():
#                         result = run_single_tree_experiment(dataset_for_exp, model=copy.deepcopy(model),
#                                                             check_diagnosis=True, faulty_nodes=[node])
#                     result["size"] = sizes[1]
#                     result["dataset"] = dataset.name
#                     result["depth"] = tree_rep[node]['depth']
#                     result["samples in node"] = model.tree_.n_node_samples[node]
#                     result["change severity"] = severity[change]
#                     if change in change_types:
#                         change = change_types[change]
#                     result["change type"] = change
#                     result["feature type"] = type_of_feature
#                     result["number of faulty nodes"] = 1
#                     result["feature in path"] = feature_in_path
#                     result["model accuracy - no drift - after"] = accuracy_after_no_drift
#                     result["model accuracy - no drift - test"] = accuracy_test_no_drift
#                     all_results.append(result)
#
#     # write results to excel
#     file_name = f"results/barinel_run_{date_time}.xlsx"
#     print(file_name)
#     workbook = xlsxwriter.Workbook(file_name)
#     worksheet = workbook.add_worksheet()
#     # write headers
#     dict_example = all_results[0]
#     index_col = {}
#     col_num = 0
#     for key in dict_example.keys():
#         worksheet.write(0, col_num, key)
#         index_col[key] = col_num
#         col_num += 1
#     # write values
#     row_num = 1
#     for dict_res in all_results:
#         for key, value in dict_res.items():
#             if type(value) in (list, set, dict):
#                 value = str(value)
#             col_num = index_col[key]
#             try:
#                 worksheet.write(row_num, col_num, value)
#             except TypeError:
#                 print(f"problem with key: '{key}', value: {value}")
#         row_num += 1
#     workbook.close()
#
#     print("DONE")
