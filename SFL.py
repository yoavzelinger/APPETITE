from sfl.Diagnoser.diagnoserUtils import write_json_planning_file, readPlanningFile

THRESHOLD = 0.1
ONLY_POSITIVE = True
MATRIX_FILE_PATH = 'matrix_for_SFL1'

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

    if result == 1: # classification is wrong
        conflicts.add(tuple(node_index))

    return test_detail, conflicts

def build_SFL_matrix_Nodes(model, samples, data_set_name):
    data_x, prediction, labels = samples
    number_of_samples = len(data_x)
    initial_tests = list(map(lambda x: f'T{x}', range(number_of_samples))) # test for each sample

    number_of_nodes = model.tree_.node_count
    components = list(map(lambda x: (x, f'node{x}'), range(number_of_nodes)))
    priors = [1/number_of_nodes]*number_of_nodes # equal prior probability to all nodes

    node_indicator = model.decision_path(data_x)  # all paths
    all_test_details = list()
    conflicts = set()
    for sample_id in range(number_of_samples):
        test_detail, conflicts = get_matrix_entrance(sample_id, samples, node_indicator, conflicts)
        all_test_details.append(test_detail)

    build_SFL_matrix(data_set_name, components, priors, initial_tests, all_test_details)
    print("list of conflicts: {}".format(conflicts))
