from datetime import datetime
import pandas as pd
import numpy as np
import copy
import shap

from DataSet import DataSet
from ResultsToExcel import write_to_excel
from Test2 import feature_order_from_node, filter_features, get_result_dict
from buildModel import build_model, map_tree, prune_tree, print_tree_rules, build_tree_for_exp


def calculate_shap2_all_nodes(model, tree_rep, samples, feature_names):
    node_count = model.tree_.node_count
    shap_values = list()

    for node in range(node_count):
        shap_val = calculate_shap2_node(node, model, tree_rep, samples, feature_names)
        shap_values.append(shap_val)

    return np.array(shap_values)  # size = [noes, samples]

def calculate_shap2_node(node, model1, tree_rep, samples, feature_names):
    # create a copy of the model and the dataset
    model = copy.deepcopy(model1)
    dataset = samples.copy()

    # if node pruned - return zeros
    if "condition" not in tree_rep[node].keys() or node == 0:
        n_samples = samples.shape[0]
        return np.zeros(n_samples)

    # # if a leaf - return zeros
    # if tree_rep[node]["left"] == -1:
    #     n_samples = samples.shape[0]
    #     return np.zeros(n_samples)

    # create a binary feature for node
    conditions = tree_rep[node]["condition"]
    feature_num = tree_rep[node]["feature"]
    feature = list(dataset.columns)[feature_num]
    indexes_filtered = (dataset[feature] > 0) | (dataset[feature] <= 0)
    for cond in conditions:
        f = cond["feature"]
        sign = cond["sign"]
        thresh = cond["thresh"]
        feature_name = feature_names[int(f)]
        if sign == ">":
            cond_filtered = dataset[feature_name] > thresh
        else:  # <=
            cond_filtered = dataset[feature_name] <= thresh
        indexes_filtered = cond_filtered & indexes_filtered

    # add a binary feature for the node to the df
    dataset["PassedInNode"] = indexes_filtered
    dataset["PassedInNode"] = dataset["PassedInNode"].astype(int)
    dataset["NegativePassedInNode"] = -dataset["PassedInNode"]

    # change the feature and the threshold in the model
    n_features = model.n_features_ + 2
    parent = tree_rep[node]["parent"]
    # model.tree_.feature[parent] = n_features - 1
    # model.tree_.threshold[parent] = 0.5
    if tree_rep[node]["type"] == "left":
        model.tree_.feature[parent] = n_features - 1
        model.tree_.threshold[parent] = -0.5
    else:  # right child
        model.tree_.feature[parent] = n_features - 2
        model.tree_.threshold[parent] = 0.5

    model.n_features_in_ = n_features
    model.n_features_ = n_features
    model.max_features_ = n_features
    model.tree_.n_features = n_features

    # print modified model
    # fs = feature_names + ["PassedInNode", "NegativePassedInNode"]
    # print_tree_rules(model, fs)

    # run shap for the data - extract shap for the new feature
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(dataset)
    node_shap = shap_values[0][:,n_features-1]

    return node_shap

def test_explainability():
    size = (0.7, 0.1, 0.2)
    time_stamp = datetime.now()
    to_sum = True

    date_time = time_stamp.strftime("%d-%m-%Y__%H-%M-%S")
    excel_name = f"PARENT fixed - explainabilityV2Check_toSum-{to_sum}_{date_time}"

    all_datasets = pd.read_csv('data/all_datasets.csv', index_col=0)

    big_trees = ["annealing", "car", "caradiotocography10clases", "image-segmentation", "mfeat-karhunen",
                 "molec-biol-splice", "socmob", "soybean", "statlog-image", "synthetic-control", "tic-tac-toe",
                 "wall-following"]

    all_results = []
    for index, row in all_datasets.iterrows():
        if row["name"] in big_trees:
            continue
        # if row["name"] not in ["blood"]:
        #     continue
        # if index > 10:  # use for testing
        #     break

        dataset = DataSet(row["path"].replace("\\", "/"), "diagnosis_check", None, None, size, name=row["name"],
                          to_shuffle=True)

        # build tree
        model, tree_rep = build_tree_for_exp(dataset)

        test_start = len(dataset.data) - dataset.test_size
        test_set = dataset.data.iloc[test_start: -1].copy()
        test_set_x = test_set[dataset.features]
        test_set_y = test_set[dataset.target]
        prediction = model.predict(test_set_x)
        n_samples = test_set_x.shape[0]
        n_features = test_set_x.shape[1]
        feature_names = dataset.features

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_set_x)

        # chack node shap v2
        total_regular_shap = 0
        node_shap_all = calculate_shap2_all_nodes(model, tree_rep, test_set_x, feature_names)

        # check individual shap
        for i in range(n_samples):
            node_sample_shap = node_shap_all[:,i]
            FI_nodes = feature_order_from_node(node_sample_shap, model, n_features, to_sum)

            regular_shap = shap_values[int(prediction[i])][i]  # takes shap value for the predicted
            total_regular_shap += np.abs(regular_shap)
            FI_shap = np.argsort(-np.array(np.abs(regular_shap)))
            FI_shap = filter_features(FI_shap, model)

            results = get_result_dict(row['name'], i, node_sample_shap.tolist(), FI_nodes, regular_shap.tolist(), FI_shap)
            all_results.append(results)

        # check global shap
        total_node_shap = np.sum(np.abs(node_shap_all), axis=1)
        global_node_shap = total_node_shap / n_samples
        FI_nodes = feature_order_from_node(global_node_shap, model, n_features, to_sum)

        global_regular_shap = total_regular_shap / n_samples
        FI_shap = np.argsort(-np.array(np.abs(global_regular_shap)))
        FI_shap = filter_features(FI_shap, model)

        results = get_result_dict(row['name'], -1, global_node_shap.tolist(), FI_nodes, global_regular_shap.tolist(),
                                  FI_shap)
        all_results.append(results)

    write_to_excel(all_results, excel_name)
    print("DONE")

if __name__ == '__main__':
    # dataset = DataSet("data/Classification_Datasets/breast-cancer-wisc-diag.csv", "diagnosis_check", None, None,
    #                    (0.7, 0.1, 0.2), name="breast-cancer-wisc-diag.csv", to_shuffle=True)
    #
    # # build tree
    # concept_size = dataset.before_size
    # target = dataset.target
    # feature_types = dataset.feature_types
    # train = dataset.data.iloc[0:int(0.9 * concept_size)].copy()
    # validation = dataset.data.iloc[int(0.9 * concept_size):concept_size].copy()
    # model = build_model(train, dataset.features, dataset.target, val_data=validation)
    # tree_rep = map_tree(model)
    # model = prune_tree(model, tree_rep)
    # print("TREE:")
    # print_tree_rules(model, dataset.features)
    # tree_rep = map_tree(model)
    #
    # test = dataset.data.iloc[concept_size:-1].copy()
    # test_x = test[dataset.features]
    # feature_names = dataset.features
    #
    # shap = calculate_shap2_all_nodes(model, tree_rep, test_x, feature_names)
    # print(f"Nodes shap all - {shap}")

    test_explainability()
    pass
