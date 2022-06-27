import copy
import numpy as np
from DataSet import DataSet
from SHAP import applySHAP
from buildModel import build_model
from sklearn import metrics
from SFL import build_SFL_matrix_SHAP, get_diagnosis, build_SFL_matrix_Nodes, get_diagnosis_nodes
from updateModel import *
#from updateModel import tree_to_code, print_tree_rules, change_tree_threshold, find_nodes_threshold_from_diagnosis, change_tree_selection, change_nodes_threshold, get_parents
from statistics import mean

#dataset = DataSet("data/hyperplane8.arff", "abrupt", "output", 1000)
#dataset = DataSet("data/mixed_1010_abrupto.csv", "abrupt", "class", 10000)
#print(type(dataset.data))

all_datasets = [
    DataSet("data/hyperplane1.arff", "abrupt", "output", 1000, ["numeric"]*10),
    DataSet("data/hyperplane2.arff", "abrupt", "output", 1000, ["numeric"]*10),
    DataSet("data/hyperplane3.arff", "abrupt", "output", 1000, ["numeric"]*10),
    DataSet("data/hyperplane4.arff", "abrupt", "output", 1000, ["numeric"]*10),
    DataSet("data/hyperplane5.arff", "abrupt", "output", 1000, ["numeric"]*10),
    DataSet("data/hyperplane6.arff", "abrupt", "output", 1000, ["numeric"]*10),
    DataSet("data/hyperplane7.arff", "abrupt", "output", 1000, ["numeric"]*10),
    DataSet("data/hyperplane8.arff", "abrupt", "output", 1000, ["numeric"]*10),
    DataSet("data/hyperplane9.arff", "abrupt", "output", 1000, ["numeric"]*10),
    DataSet("data/rt_2563789698568873_abrupto.csv", "abrupt", "class", 10000, ["numeric"]*2),
    DataSet("data/sea_0123_abrupto_noise_0.2.csv", "abrupt", "class", 10000, ["numeric"]*3),
    DataSet("data/mixed_0101_abrupto.csv", "abrupt", "class", 10000, ["binary"]*2+["numeric"]*2),
    DataSet("data/mixed_1010_abrupto.csv", "abrupt", "class", 10000, ["binary"]*2+["numeric"]*2),
    DataSet("data/stagger_2102_abrupto.csv", "abrupt", "class", 10000, ["categorical"]*4)
]

SIZE = -1
NEW_DATA_SIZE = -1

def feature_diff_after_concept_drift(data_set, drift_size, new_data_size):
    diff = dict()
    i = 0
    for feature in data_set.features:
        before = data_set.data[feature].iloc[0:drift_size]
        before_avg = before.mean()
        after = data_set.data[feature].iloc[drift_size: drift_size + new_data_size]
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
    features_diff = feature_diff_after_concept_drift(dataset, SIZE, NEW_DATA_SIZE)
    nodes, thresholds = find_nodes_threshold_from_diagnosis(model, diagnosis, features_diff)
    model_to_fix = copy.deepcopy(model)
    fixed_model = change_tree_threshold(model_to_fix, nodes, thresholds)
    return fixed_model

def diagnose_Nodes(model, dataset, new_data):
    nodes = model.tree_.node_count
    print("number of nodes: {}".format(nodes))

    # get diagnosis - amir's code
    ## build_SFL_matrix_Nodes(model, new_data, dataset.name)
    ## diagnosis = get_diagnosis()
    ## first_diagnosis = diagnosis[0].diagnosis

    # get diagnosis - avi's code
    (diagnoses, probabilities), BAD_SAMPLES = get_diagnosis_nodes(model, new_data)
    print("diagnoses: {}".format(diagnoses))
    print("probabilities: {}".format(probabilities))
    first_diagnosis = diagnoses[0]

    return first_diagnosis, BAD_SAMPLES

def fix_nodes_binary(model, diagnosis):
    # fix model - Nodes, change selection (right <--> left)
    model_to_fix = copy.deepcopy(model)
    fixed_model = change_tree_selection(model_to_fix, diagnosis)
    return fixed_model

def fix_nodes_numeric(model, diagnosis, dataset):
    # fix model - Nodes, change node's thresholds
    model_to_fix = copy.deepcopy(model)
    features_diff = feature_diff_after_concept_drift(dataset, SIZE, NEW_DATA_SIZE)
    fixed_model = change_nodes_threshold(model_to_fix, diagnosis, features_diff)
    return fixed_model

def fix_nodes_by_type(model, diagnosis, dataset):
    # fix model - Nodes, change selection or threshold
    model_to_fix = copy.deepcopy(model)
    features_diff = feature_diff_after_concept_drift(dataset, SIZE, NEW_DATA_SIZE)
    fixed_model = change_nodes_by_type(model_to_fix, diagnosis, dataset.feature_types, features_diff)
    return fixed_model

def run_single_tree_experiment(dataset):
    global SIZE, NEW_DATA_SIZE
    SIZE = int(dataset.batch_size * 1)
    #NEW_DATA_SIZE = int(0.1 * SIZE)
    NEW_DATA_SIZE = 100
    model = build_model(dataset.data.iloc[0:int(0.9*SIZE)], dataset.features, dataset.target)

    # check model accuracy on data before the drift
    test_data = dataset.data.iloc[int(0.9*SIZE): SIZE]
    test_data_x = test_data[dataset.features]
    prediction = model.predict(test_data_x)
    test_data_y = test_data[dataset.target]
    accuracy = metrics.accuracy_score(test_data_y, prediction)
    print("Accuracy of original model on data BEFORE concept drift:", accuracy)

    # check model accuracy on data after concept drift
    new_data = dataset.data.iloc[SIZE: SIZE + NEW_DATA_SIZE]
    new_data_x = new_data[dataset.features]
    prediction = model.predict(new_data_x)
    new_data_y = new_data[dataset.target]
    accuracy = metrics.accuracy_score(new_data_y, prediction)
    print("Accuracy of original model on data AFTER concept drift:", accuracy)

    #print("TREE:")
    #print_tree_rules(model, dataset.features)

    # RUN ALGORITHM
    samples = (new_data_x, prediction, new_data_y)
    # build_SFL_matrix_Nodes(model, samples, dataset.name)
    diagnosis, BAD_SAMPLES = diagnose_Nodes(model, dataset, samples)
    print(f"best diagnosis: {diagnosis}")
    # fixed_model = fix_nodes_binary(model, diagnosis)
    # fixed_model = fix_nodes_numeric(model, diagnosis, dataset)
    fixed_model = fix_nodes_by_type(model, diagnosis, dataset)

    # print("FIXED TREE:")
    # print_tree_rules(fixed_model, dataset.features)

    """
    # run algorithm
    diagnosis = diagnose_SHAP(model, dataset, new_data)
    fixed_model = fix_SHAP(model, diagnosis, dataset)

    # tree_to_code(model, dataset.features)
    # print_tree_rules(model, dataset.features)
    """
    print("--- new data accuracy ---")
    print("Accuracy of original model on data after concept drift:", accuracy)

    prediction = fixed_model.predict(new_data_x)
    accuracy = metrics.accuracy_score(new_data_y, prediction)
    print("Accuracy of Fixed model on data after concept drift:", accuracy)

    # TEST performances
    print("--- test data accuracy ---")
    test_set = dataset.data.iloc[SIZE + NEW_DATA_SIZE: SIZE + 2 * NEW_DATA_SIZE]
    test_set_x = test_set[dataset.features]
    test_set_y = test_set[dataset.target]

    # check original model on the new data
    prediction1 = model.predict(test_set_x)
    accuracy = metrics.accuracy_score(test_set_y, prediction1)
    print("Accuracy of the Original model on test data:", accuracy)

    # train a new model with data before and after drift
    model_all = build_model(dataset.data.iloc[0:SIZE + NEW_DATA_SIZE], dataset.features, dataset.target)
    prediction2 = model_all.predict(test_set_x)
    accuracy = metrics.accuracy_score(test_set_y, prediction2)
    print("Accuracy of a New model (before & after) on test data:", accuracy)

    # train a new model on data after drift
    model_after = build_model(dataset.data.iloc[SIZE:SIZE + NEW_DATA_SIZE], dataset.features, dataset.target)
    prediction4 = model_after.predict(test_set_x)
    accuracy = metrics.accuracy_score(test_set_y, prediction4)
    print("Accuracy of a New model (only after) on test data:", accuracy)

    # check the fixed model
    prediction3 = fixed_model.predict(test_set_x)
    accuracy = metrics.accuracy_score(test_set_y, prediction3)
    print("Accuracy of Fixed model on test data:", accuracy)

    print("--- misclassified (new) data accuracy ---")
    bad_samples_indexes = np.array(BAD_SAMPLES) + SIZE
    bad_samples = dataset.data.iloc[bad_samples_indexes]
    print(f"number of bad somples: {len(bad_samples)}")
    bad_samples_x = bad_samples[dataset.features]
    prediction_bad = model.predict(bad_samples_x)
    bad_samples_y = bad_samples[dataset.target]
    accuracy = metrics.accuracy_score(bad_samples_y, prediction_bad)
    print("Accuracy of Fixed model on BAD samples only:", accuracy)

    """
    print("Data after the drift")
    print(new_data)
    print("Test set")
    #print(test_set)
    """

if __name__ == '__main__':
    for data in all_datasets[9:]:
        print(f"#### Experiment of dataset: {data.name} ####")
        run_single_tree_experiment(data)
        print("-----------------------------------------------------------------------------------------")
