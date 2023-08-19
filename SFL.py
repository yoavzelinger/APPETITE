from sfl.Diagnoser.diagnoserUtils import write_json_planning_file, readPlanningFile
import numpy as np
from Barinel import calculate_diagnoses_and_probabilities_barinel_shaked
from NodeSHAP import calculate_tree_values, calculate_shap_all_nodes
from SingleFault import diagnose_single_fault
from buildModel import calculate_error, calculate_left_right_ratio

THRESHOLD = 0.1
ONLY_POSITIVE = True
MATRIX_FILE_PATH = 'matrix_for_SFL1'
PARENTS = dict()
epsilon = np.finfo(np.float64).eps

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def build_SFL_matrix(description, components_names, priors, initial_tests, test_details):
    with open(MATRIX_FILE_PATH, 'w') as f:
        f.write('[Description]\n')
        f.write('{}\n'.format(description))

        f.write('[Components names]\n')
        f.write('{}\n'.format(components_names))

        f.write('[Priors]\n')
        f.write('{}\n'.format(priors))

        f.write('[Bugs]\n')  # always empty - irrelevant
        f.write('[]\n')

        f.write('[InitialTests]\n')
        f.write('{}\n'.format(initial_tests))

        f.write('[TestDetails]\n')  # list of tests
        for test in test_details:
            f.write('{}\n'.format(test))

def build_SFL_matrix_SHAP(features, shap_values, prediction, labels, data_set_name):
    number_of_samples = len(prediction)
    initial_tests = list(map(lambda x: f'T{x}', range(number_of_samples)))  # test for each sample

    positive_components = list(map(lambda x: (2*x, f'{features[x]} positive'), range(len(features))))
    negative_components = list(map(lambda x: (2*x+1, f'{features[x]} negative'), range(len(features))))
    components_names = [*positive_components, *negative_components]
    priors = [1/(len(components_names))]*(len(components_names))  # equal prior probability to all components

    conflicts = set()
    all_test_details = list()
    for i in range(number_of_samples):  # i is sample index
        result = 0 if prediction[i] == labels.values[i] else 1  # 1 if there is an error in prediction
        components = list()
        shap = shap_values[int(prediction[i])][i]  # takes shap value for the predicted
        # TODO: calculate components using filters instead of for
        for j in range(len(shap)): # j is feature index
            value = shap[j]
            if ONLY_POSITIVE:
                if value >= THRESHOLD:
                    component_id = 2 * j
                    components.append(component_id)
            elif abs(value) >= THRESHOLD:  # feature appears as a component
                component_id = 2 * j if value > 0 else 2 * j + 1
                components.append(component_id)

        test_detail = f'T{i};{components};{result}'
        all_test_details.append(test_detail)
        if result == 1:  # classification is wrong
            conflicts.add(tuple(components))

    build_SFL_matrix(data_set_name, components_names, priors, initial_tests, all_test_details)
    print("list of conflicts: {}".format(conflicts))

def get_diagnosis():
    ei = readPlanningFile(MATRIX_FILE_PATH)
    ei.diagnose()
    return ei.diagnoses

def get_matrix_entrance(sample_id, samples, node_indicator, conflicts):
    data_x, prediction, labels = samples
    node_index = node_indicator.indices[            # extract the relevant path for sample_id
                 node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                 ].tolist()
    result = 0 if prediction[sample_id] == labels.values[sample_id] else 1  # 1 if there is an error in prediction
    test_detail = f"T{sample_id};{node_index};{result}"

    if result == 1:  # classification is wrong
        conflicts.add(tuple(node_index))

    return test_detail, conflicts

def build_SFL_matrix_Nodes(model, samples, data_set_name):
    data_x, prediction, labels = samples
    number_of_samples = len(data_x)
    initial_tests = list(map(lambda x: f'T{x}', range(number_of_samples))) # test for each sample

    number_of_nodes = model.tree_.node_count
    components = list(map(lambda x: (x, f'node{x}'), range(number_of_nodes)))
    priors = [1/number_of_nodes]*number_of_nodes  # equal prior probability to all nodes

    node_indicator = model.decision_path(data_x)  # all paths
    all_test_details = list()
    conflicts = set()
    for sample_id in range(number_of_samples):
        test_detail, conflicts = get_matrix_entrance(sample_id, samples, node_indicator, conflicts)
        all_test_details.append(test_detail)

    build_SFL_matrix(data_set_name, components, priors, initial_tests, all_test_details)
    print("list of conflicts: {}".format(conflicts))

def get_SFL_for_diagnosis_nodes(model, samples, model_rep):
    BAD_SAMPLES = list()
    data_x, prediction, labels = samples
    number_of_samples = len(data_x)
    number_of_nodes = model.tree_.node_count

    # initialize spectra and error vector
    error_vector = np.zeros(number_of_samples)
    spectra = np.zeros((number_of_samples, number_of_nodes))

    node_indicator = model.decision_path(data_x)  # get paths for all samples
    conflicts = set()
    errors = 0
    for sample_id in range(number_of_samples):
        node_index = node_indicator.indices[  # extract the relevant path for sample_id
                     node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                     ].tolist()
        for node_id in node_index:
            # set as a component in test
            spectra[sample_id][node_id] = 1
        if prediction[sample_id] != labels.values[sample_id]:  # test result is "fail"
            error_vector[sample_id] = 1
            errors += 1
            conflicts.add(tuple(node_index))
            BAD_SAMPLES.append(sample_id)

    print(f"Conflicts: {conflicts}")
    print(f"Number of misclassified samples: {errors}")
    return BAD_SAMPLES, spectra, error_vector, conflicts

def get_prior_probs_depth(model_rep, number_of_nodes):
    # define prior vector
    priors = np.ones(number_of_nodes) * 0.99
    depth = [model_rep[node]["depth"] if "parent" in model_rep[node] else 0 for node in range(number_of_nodes)]
    # depth = [model_rep[node]["depth"] if node in model_rep else 0 for node in range(number_of_nodes)]
    max_depth = max(depth)
    # priors = [
    #     0.99**(max_depth - depth[node])
    #     if node in model_rep and model_rep[node]["left"] != -1
    #     else 0.99**((max_depth - depth[node])*4)
    #     for node in range(number_of_nodes)]
    # priors = [  # BEST FOR: barinel single node
    #     0.01 * (depth[node]+1)
    #     if node in model_rep and model_rep[node]["left"] != -1
    #     else 0.01 * (depth[node]+1)/4
    #     for node in range(number_of_nodes)]
    # priors = [
    #     0.1 / (max_depth - depth[node] + 1)
    #     if node in model_rep and model_rep[node]["left"] != -1
    #     else 0.1 / (4*(max_depth - depth[node] + 1))
    #     for node in range(number_of_nodes)]
    priors = [  # BEST FOR: barinel original
        1 - ((max_depth - depth[node] + 1) / (max_depth + 2))
        if node in model_rep and model_rep[node]["left"] != -1
        else (1 - ((max_depth - depth[node] + 1) / (max_depth + 2))) / 4
        for node in range(number_of_nodes)]
    priors = np.array(priors)
    return priors

def get_prior_probs_left_right(model_rep, spectra):
    # define prior vector
    number_of_nodes = spectra.shape[1]
    priors = calculate_left_right_diff(spectra, model_rep)
    # priors = softmax(priors)
    priors = np.array(priors)
    return priors

def get_diagnosis_barinel(spectra, error_vector, priors):
    diagnoses, probabilities = calculate_diagnoses_and_probabilities_barinel_shaked(spectra, error_vector, priors)
    return diagnoses, probabilities

def get_diagnosis_single_fault(spectra, error_vector, similarity_method,priors=None):
    diagnoses, probabilities = diagnose_single_fault(spectra, error_vector, similarity_method, priors)
    return diagnoses, probabilities

def calculate_nodes_error(spectra, error_vector):
    participation = spectra.sum(axis=0)
    errors = (spectra * (error_vector.reshape(-1,1))).sum(axis=0)
    error_rate = errors / (participation + epsilon)
    return error_rate

def calculate_left_right_diff(spectra, model_rep):
    n_nodes = spectra.shape[1]
    left_right_dict = calculate_left_right_ratio(model_rep)
    original_ratio = np.zeros(n_nodes)
    for node, ratio in left_right_dict.items():
        original_ratio[node] = ratio

    participation = spectra.sum(axis=0)
    left_right_current = np.zeros(n_nodes)
    nodes_to_check = [0]
    while len(nodes_to_check) > 0:
        node = nodes_to_check.pop(0)
        left = model_rep[node]["left"]
        right = model_rep[node]["right"]

        if left != -1:  # not a leaf
            total = participation[node] + epsilon
            went_left = participation[left]
            left_right_current[node] = went_left / total
            nodes_to_check.append(left)
            nodes_to_check.append(right)
        else:
            left_right_current[node] = -1

    diff_ratio = np.absolute(left_right_current - original_ratio)
    return diff_ratio

def get_diagnosis_error_rate(spectra, error_vector, model_rep):
    n_nodes = spectra.shape[1]
    original_errors_dict = calculate_error(model_rep)
    original_errors = np.zeros(n_nodes)
    for node, error_rate in original_errors_dict.items():
        original_errors[node] = error_rate

    current_errors = calculate_nodes_error(spectra, error_vector)
    for node in range(n_nodes):
        cur_error_rate = current_errors[node]
        if np.isnan(cur_error_rate):
            current_errors[node] = 0

    diff_error = current_errors - original_errors
    #diff_error = (current_errors - original_errors) / (original_errors + epsilon)
    # diff_error = (current_errors - original_errors) * (np.power(original_errors + epsilon, 0.9)/(original_errors + epsilon))
    d_order = np.argsort(-diff_error)
    diagnoses = list(map(int, d_order))
    rank = diff_error[d_order]
    return diagnoses, rank

def get_diagnosis_left_right(spectra, error_vector, model_rep):
    diff_ratio = calculate_left_right_diff(spectra, model_rep)
    d_order = np.argsort(-diff_ratio)
    diagnoses = list(map(int, d_order))
    rank = diff_ratio[d_order]
    return diagnoses, rank

def get_diagnosis_node_shap(samples, model_rep, f="confident"):
    shap_values = get_prior_probs_node_shap(samples, model_rep, f)
    d_order = np.argsort(-shap_values)
    diagnoses = list(map(int, d_order))
    rank = shap_values[d_order]
    return diagnoses, rank

def get_prior_probs_node_shap(samples, model_rep, f="confident", tree_analysis=None):
    data_x, prediction, labels = samples

    if tree_analysis is None:
        # all_ans = calculate_tree_values(model_rep)
        # tree_analysis = all_ans[0]
        tree_analysis = calculate_tree_values(model_rep)

    m = 0
    i = -1
    node_count = len(model_rep) - 2
    shap_values = np.zeros(node_count)
    for index, sample in data_x.iterrows():
        i += 1
        if prediction[i] == labels[index]:  # skip samples that classified correctly
            continue
        m += 1
        shap = calculate_shap_all_nodes(model_rep, tree_analysis, sample, f)
        shap_values += np.absolute(np.array(shap))

    if m > 0:
        shap_values /= m
    else:  # no misclassified samples
        shap_values = np.ones(node_count)
    return shap_values


def shap_nodes_to_SFL(samples, model_rep, f="confident", tree_analysis=None):
    data_x, prediction, labels = samples
    n_samples = len(data_x)
    node_count = len(model_rep) - 2

    if tree_analysis is None:
        tree_analysis = calculate_tree_values(model_rep)

    # initialize spectra and error vector
    error_vector = np.zeros(n_samples)
    spectra = np.zeros((n_samples, node_count))
    BAD_SAMPLES = list()
    conflicts = set()

    errors = 0
    i = -1
    for index, sample in data_x.iterrows():
        i += 1
        # add mistakes to error vector
        if prediction[i] != labels[index]:  # skip samples that classified correctly
            error_vector[i] = 1
            errors += 1
            BAD_SAMPLES.append(i)

        # add shap to the SFL
        shap = calculate_shap_all_nodes(model_rep, tree_analysis, sample, f)
        spectra[i,:] = np.absolute(np.array(shap)) #TODO: think if we need negative values

    return BAD_SAMPLES, spectra, error_vector, conflicts



