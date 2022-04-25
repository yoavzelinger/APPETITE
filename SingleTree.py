import copy

from DataSet import DataSet
from SHAP import applySHAP
from buildModel import build_model
from sklearn import metrics
from SFL import build_SFL_matrix_SHAP, get_diagnosis, build_SFL_matrix_Nodes
from updateModel import tree_to_code, print_tree_rules, change_tree_threshold, find_nodes_threshold_from_diagnosis
from statistics import mean

dataset = DataSet("data/hyperplane1.arff", "abrupt", "output", 1000)
#dataset = DataSet("data/stagger_2102_abrupto.csv", "abrupt", "class", 10000)

SIZE = dataset.batch_size

def feature_diff_after_concept_drift(data_set, drift_size):
    diff = dict()
    i = 0
    for feature in data_set.features:
        before = data_set.data[feature].iloc[0:drift_size]
        before_avg = before.mean()
        after = data_set.data[feature].iloc[drift_size: int(1.1*drift_size)]
        after_avg = after.mean()
        diff[i] = before_avg - after_avg
        i += 1
    return diff

def diagnose_SHAP(model, dataset, new_data):
    # diagnose model - SHAP
    shap_values = applySHAP(dataset.features, new_data, model)
    new_data_y = new_data[dataset.target]
    build_SFL_matrix_SHAP(dataset.features, shap_values, prediction, new_data_y, dataset.name)
    diagnosis = get_diagnosis()
    print("diagnosis: {}".format(diagnosis))
    # TODO: rate diagnosis
    first_diagnosis = diagnosis[0].diagnosis
    return first_diagnosis

def fix_SHAP(model, diagnosis, dataset):
    # fix model - SHAP
    features_diff = feature_diff_after_concept_drift(dataset, SIZE)
    nodes, thresholds = find_nodes_threshold_from_diagnosis(model, diagnosis, features_diff)
    model_to_fix = copy.deepcopy(model)
    fixed_model = change_tree_threshold(model_to_fix, nodes, thresholds)
    return fixed_model

def diagnose_Nodes(model, dataset, new_data):
    nodes = model.tree_.node_count
    print("number of nodes: {}".format(nodes))
    build_SFL_matrix_Nodes(model, new_data, dataset.name)
    diagnosis = get_diagnosis()
    print("diagnosis: {}".format(diagnosis))
    # TODO: rate diagnosis
    first_diagnosis = diagnosis[0].diagnosis
    return first_diagnosis

if __name__ == '__main__':
    model = build_model(dataset.data.iloc[0:SIZE], dataset.features, dataset.target)

    # check model accuracy on data after concept drift
    new_data = dataset.data.iloc[SIZE: int(1.1*SIZE)]
    new_data_x = new_data[dataset.features]
    prediction = model.predict(new_data_x)
    new_data_y = new_data[dataset.target]
    accuracy = metrics.accuracy_score(new_data_y, prediction)
    print("Accuracy of original model on data after concept drift:", accuracy)

    samples = (new_data_x, prediction, new_data_y)
    # build_SFL_matrix_Nodes(model, samples, dataset.name)
    diagnosis = diagnose_Nodes(model, dataset, samples)
    print(diagnosis)

    """
    # run algorithm
    diagnosis = diagnose_SHAP(model, dataset, new_data)
    fixed_model = fix_SHAP(model, diagnosis, dataset)

    # tree_to_code(model, dataset.features)
    # print_tree_rules(model, dataset.features)

    # TEST performances
    test_set = dataset.data.iloc[int(1.1 * SIZE): int(1.2 * SIZE)]
    test_set_x = test_set[dataset.features]
    test_set_y = test_set[dataset.target]

    # check original model on the new data
    prediction1 = model.predict(test_set_x)
    accuracy = metrics.accuracy_score(test_set_y, prediction1)
    print("Accuracy of ORIGINAL model:", accuracy)

    # train a new model
    model_all = build_model(dataset.data.iloc[0:int(1.1 * SIZE)], dataset.features, dataset.target)
    prediction2 = model_all.predict(test_set_x)
    accuracy = metrics.accuracy_score(test_set_y, prediction2)
    print("Accuracy of a model trained with some data after concept drift:", accuracy)

    # check the fixed model
    prediction3 = fixed_model.predict(test_set_x)
    accuracy = metrics.accuracy_score(test_set_y, prediction3)
    print("Accuracy of FIXED model:", accuracy)
    """