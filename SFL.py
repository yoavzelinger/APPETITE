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
    with open('matrix_for_SFL', 'w') as f:
        f.write('[Description]\n')
        f.write('{}\n'.format(data_set_name))

        f.write('[Components names]\n')
        f.write('[')
        i = 0
        for i in range(len(features)):
            if i > 0:
                f.write(', ')
            f.write('({}, \'{} positive\')'.format(2*i, features[i]))
            f.write(', ')
            f.write('({}, \'{} negative\')'.format(2*i+1, features[i]))
        f.write(']\n')

        f.write('[Priors]\n')
        f.write('[')
        i = 0
        for i in range(len(features)*2):
            if i > 0:
                f.write(', ')
            f.write('0.1')
        f.write(']\n')

        f.write('[Bugs]\n')
        f.write('[]\n')

        f.write('[InitialTests]\n')
        f.write('[')
        for i in range(len(prediction)):
            if i > 0:
                f.write(', ')
            f.write('\'T{}\''.format(i))
        f.write(']\n')

        conflicts = set()
        f.write('[TestDetails]\n')
        for i in range(len(prediction)):
            result = 0 if prediction[i] == labels.values[i] else 1  # 1 if there is an error in prediction
            components = list()
            shap = shap_values[int(prediction[i])][i]  # takes shap value for the predicted
            for j in range(len(shap)):
                value = shap[j]
                if ONLY_POSITIVE:
                    if value >= THRESHOLD:
                        component_id = 2 * j
                        components.append(component_id)
                elif abs(value) >= THRESHOLD:  # feature appears as a component
                    component_id = 2*j if value > 0 else 2*j+1
                    components.append(component_id)
            f.write('T{};{};{}\n'.format(i, components, result))
            if result == 1:
                conflicts.add(tuple(components))
        print("list of conflicts: {}".format(conflicts))

def get_diagnosis():
    ei = readPlanningFile(r"matrix_for_SFL")
    ei.diagnose()
    return ei.diagnoses

def get_matrix_entrance(sample_id, samples, node_indicator):
    data_x, prediction, labels = samples
    node_index = node_indicator.indices[            # extract the relevant path for sample_id
                 node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                 ].tolist()
    result = 0 if prediction[sample_id] == labels.values[sample_id] else 1  # 1 if there is an error in prediction
    test_detail = "T{};{};{}".format(sample_id, node_index, result)  # test number;[components];test result
    return test_detail

def build_SFL_matrix_Nodes(model, samples, data_set_name):
    data_x, prediction, labels = samples
    number_of_samples = len(data_x)

    number_of_nodes = model.tree_.node_count
    components = list(range(number_of_nodes))

    # TODO: priors + initial tests
    priors = list()
    initial_tests = list()

    node_indicator = model.decision_path(data_x)  # all paths
    all_test_details = list()
    for sample_id in range(number_of_samples):
        test_detail = get_matrix_entrance(sample_id, samples, node_indicator)
        all_test_details.append(test_detail)

    build_SFL_matrix(data_set_name, components, priors, initial_tests, all_test_details)
    return
