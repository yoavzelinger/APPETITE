from datetime import datetime
import pickle
import shap
import numpy as np
import math

from DataSet import DataSet
import pandas as pd

from NodeSHAP import calculate_shap_all_nodes
from ResultsToExcel import write_to_excel
from buildModel import build_tree_for_exp

def map_nodes_to_feature(model):
    n_nodes = model.tree_.node_count
    node_features = []
    features = set()
    for node in range(n_nodes):
        f = model.tree_.feature[node]
        node_features.append(f)
        if f != -2:
            features.add(f)
    return node_features, features

def filter_features(FI_shap):  # remove from FI features that are not on the tree
    _, features = map_nodes_to_feature(model)
    filtered_FI = []
    for f in FI_shap:
        if f in features:
            filtered_FI.append(f)
    return filtered_FI

def feature_order_from_node(shap_vals, model, n_features, to_sum=True):
    features_by_node, features = map_nodes_to_feature(model)

    if to_sum:
        f_importance = np.zeros(n_features)
        n_nodes = len(shap_vals)
        for node in range(n_nodes):
            f = features_by_node[node]
            if f == -2:
                continue
            f_importance[f] += shap_vals[node]

        feature_order = np.argsort(-np.array(f_importance)).tolist()

    else:
        node_order = np.argsort(-np.array(shap_vals))
        feature_order = list(map(lambda node: features_by_node[node], node_order.tolist()))
        feature_order = list(dict.fromkeys(feature_order))  # not duplicated, but retain the same order
        # # add features that are not in the tree
        # for f in range(n_features):
        #     if f not in feature_order:
        #         feature_order.append(f)
        feature_order.remove(-2)  # remove leaf indicator -2

    feature_order = filter_features(feature_order)
    return feature_order

def calculate_kendalls_tau(shap_original_order, node_shap_order):
    assert len(shap_original_order) == len(node_shap_order)
    n_features = len(shap_original_order)
    if n_features < 2:
        return 1

    def nCr(n, r):
        f = math.factorial
        return f(n) // f(r) // f(n - r)
    n_pairs = nCr(n_features, 2)

    concordat_pairs, discordant_pairs = 0,0

    for rank1_shap in range(n_features):
        node1 = shap_original_order[rank1_shap]
        rank1_node = node_shap_order.index(node1)

        for rank2_shap in range(rank1_shap+1, n_features):
            node2 = shap_original_order[rank2_shap]
            rank2_node = node_shap_order.index(node2)

            if rank1_shap > rank2_shap and rank1_node > rank2_node:
                concordat_pairs += 1
            elif rank1_shap < rank2_shap and rank1_node < rank2_node:
                concordat_pairs += 1
            else:
                discordant_pairs += 1

    tau = (concordat_pairs - discordant_pairs) / n_pairs
    return tau

def get_result_dict(db_name, index, node_shap, FI_nodes, regular_shap, FI_shap):
    results = {}
    results["dataset"] = db_name
    results["index"] = index
    results["tree size"] = len(node_shap)
    results["#features"] = len(regular_shap)
    results["#features in tree"] = len(FI_shap)

    results["node shap"] = node_shap
    results["node shap FI"] = FI_nodes

    results["regular shap"] = regular_shap
    results["regular shap FI"] = FI_shap

    # compare
    is_same_order = np.array_equal(FI_nodes, FI_shap)
    results["is same order"] = 1 if is_same_order else 0
    if len(FI_nodes) > 0:
        is_first_same = FI_nodes[0] == FI_shap[0]
        results["is first same"] = 1 if is_first_same else 0
    else:
        results["is first same"] = -1
    if len(FI_nodes) > 1:
        is_second_same = FI_nodes[1] == FI_shap[1]
        results["is second same"] = 1 if is_second_same else 0
    else:
        results["is second same"] = -1

    tau = calculate_kendalls_tau(FI_shap, FI_nodes)
    results["Kendall's tau"] = tau

    return results

# def sort_rank(shap):
#     idx, = np.where(shap > 0)
#     non_zero = idx[np.argsort(shap[idx])]
#     all_zero, = np.where(shap == 0)
#     rank = np.concatenate((non_zero, all_zero))
#     return rank

if __name__ == '__main__':
    size = (0.7, 0.1, 0.2)
    shap_measure = "confident"
    time_stamp = datetime.now()
    to_sum = True

    date_time = time_stamp.strftime("%d-%m-%Y__%H-%M-%S")
    excel_name = f"explainabilityCheck_toSum-{to_sum}_{date_time}"

    all_datasets = pd.read_csv('data/all_datasets.csv', index_col=0)

    big_trees = ["annealing", "car", "caradiotocography10clases", "image-segmentation", "mfeat-karhunen",
                 "molec-biol-splice", "socmob", "soybean", "statlog-image", "synthetic-control", "tic-tac-toe",
                 "wall-following"]

    all_results = []
    for index, row in all_datasets.iterrows():
        if row["name"] in big_trees:
            continue
        # if row["name"] not in ["audiology-std"]:
        #     continue
        # if index > 10:  # use for testing
        #     break

        dataset = DataSet(row["path"].replace("\\", "/"), "diagnosis_check", None, None, size, name=row["name"],
                          to_shuffle=True)

        # build tree
        model, tree_rep = build_tree_for_exp(dataset)

        # SHAP - create tree analysis
        pickle_path = f"tree_analysis\\{row['name']}.pickle"
        with open(pickle_path, "rb") as file:
            tree_analysis = pickle.load(file)

        test_start = len(dataset.data) - dataset.test_size
        test_set = dataset.data.iloc[test_start: -1].copy()
        test_set_x = test_set[dataset.features]
        test_set_y = test_set[dataset.target]
        prediction = model.predict(test_set_x)
        n_samples = test_set_x.shape[0]
        n_features = test_set_x.shape[1]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_set_x)

        total_node_shap = 0
        total_node_shap_only_faulty = 0
        total_regular_shap = 0
        total_regular_shap_only_faulty = 0

        # check individual shap
        j = -1
        for index, sample in test_set_x.iterrows():
            j += 1

            node_shap = calculate_shap_all_nodes(tree_rep, tree_analysis, sample, shap_measure)
            total_node_shap += np.abs(node_shap)
            FI_nodes = feature_order_from_node(node_shap, model, n_features, to_sum)

            regular_shap = shap_values[int(prediction[j])][j]  # takes shap value for the predicted
            total_regular_shap += np.abs(regular_shap)
            FI_shap = np.argsort(-np.array(np.abs(regular_shap)))
            FI_shap = filter_features(FI_shap)
            # FI_shap = sort_rank(np.abs(regular_shap))

            results = get_result_dict(row['name'], index, node_shap.tolist(), FI_nodes, regular_shap.tolist(), FI_shap)
            all_results.append(results)

        # check global shap
        global_node_shap = total_node_shap/n_samples
        FI_nodes = feature_order_from_node(global_node_shap, model, n_features, to_sum)

        global_regular_shap = total_regular_shap / n_samples
        # FI_shap = sort_rank(np.abs(global_regular_shap))
        FI_shap = np.argsort(-np.array(np.abs(global_regular_shap)))
        FI_shap = filter_features(FI_shap)

        results = get_result_dict(row['name'], -1, global_node_shap.tolist(), FI_nodes, global_regular_shap.tolist(),
                                  FI_shap)
        all_results.append(results)

    write_to_excel(all_results, excel_name)
    print("DONE")
