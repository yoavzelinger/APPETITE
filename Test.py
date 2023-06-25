import copy
import pickle
import numpy as np

from DataSet import DataSet
from DriftSimulationRun import is_feature_in_tree, manipulate_feature
from ResultsToExcel import write_to_excel
from SingleTree import run_single_tree_experiment
from buildModel import build_model, map_tree, prune_tree, build_tree_for_exp
from NodeSHAP import calculate_tree_values, calculate_shap_all_nodes

def node_order_by_shap(shap):
    node_order = np.argsort(-np.array(shap))
    return node_order

if __name__ == '__main__':
    path = "data/Classification_Datasets/acute-inflammation.csv"
    db_name = "acute-inflammation"
    size = (0.7, 0.1, 0.2)

    similarity_measure = "prior"  # if prior so no sfl
    prior_measure = "node_shap"
    shap_measure = "confident"

    methods = {
        "SFL": similarity_measure,
        "prior": prior_measure,
        "SHAP": shap_measure
    }

    dataset = DataSet(path.replace("\\", "/"), "diagnosis_check", None, None, size, name=db_name,
                      to_shuffle=True)

    # build tree
    model, tree_rep = build_tree_for_exp(dataset)

    samples_shap = {}
    all_results = []

    # SHAP - create tree analysis
    pickle_path = f"tree_analysis\\{db_name}.pickle"
    with open(pickle_path, "rb") as file:
        tree_analysis = pickle.load(file)

    # manipulate data & run experiment
    for i in range(len(dataset.features)):
        if not is_feature_in_tree(tree_rep, i):
            continue

        feature = dataset.features[i]
        feature_type = dataset.feature_types[i]
        if feature_type == "categorical":
            n_values = dataset.data[feature].unique().size
            if n_values > 2:
                f_type = "categorical"
            else:
                f_type = "binary"
        else:
            f_type = "numeric"

        manipulated_data = manipulate_feature(feature, feature_type, dataset)

        for data, change in manipulated_data:
            test_start = len(dataset.data) - dataset.test_size
            test_end = -1  # the end of the dataset
            test_set = data.iloc[test_start: test_end].copy()
            test_set_x = test_set[dataset.features]
            test_set_y = test_set[dataset.target]
            prediction = model.predict(test_set_x)

            j = -1
            for index, sample in test_set_x.iterrows():
                j += 1
                # if prediction[j] == test_set_y[index]:  # skip samples that classified correctly
                #     continue

                results = {}
                results["feature id"] = i
                results["feature_type"] = f_type
                results["change"] = change
                results["is_misclasification?"] = 0 if prediction[j] == test_set_y[index] else 1
                results["index"] = index

                # check original sample shap
                if index in samples_shap:
                    original_shap = samples_shap[index]
                else:
                    original_sample = dataset.data.iloc[index]
                    original_shap = calculate_shap_all_nodes(tree_rep, tree_analysis, original_sample, shap_measure)
                    samples_shap[index] = original_shap
                results["original shap"] = original_shap.tolist()
                original_order = node_order_by_shap(original_shap)
                results["original order"] = original_order.tolist()

                # check manipulated sample shap
                modified_shap = calculate_shap_all_nodes(tree_rep, tree_analysis, sample, shap_measure)
                results["modified shap"] = modified_shap.tolist()
                modified_order = node_order_by_shap(modified_shap)
                results["modified order"] = modified_order.tolist()

                # compare
                is_same = np.array_equal(original_shap, modified_shap)
                results["is same shap"] = 1 if is_same else 0
                is_same_order = np.array_equal(original_order, modified_order)
                results["is same order"] = 1 if is_same_order else 0

                all_results.append(results)

    write_to_excel(all_results, "changes_test_run")
    print("DONE")
