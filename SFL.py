from sfl.Diagnoser.diagnoserUtils import write_json_planning_file, readPlanningFile

THRESHOLD = 0.15

def build_SFL_matrix(features, shap_values, prediction, labels, data_set_name):
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

        f.write('[TestDetails]\n')
        for i in range(len(prediction)):
            result = 0 if prediction[i] == labels.values[i] else 1  # 1 if there is an error in prediction
            components = list()
            shap = shap_values[int(labels.values[i])][i]  # takes shap value for the correct label
            for j in range(len(shap)):
                value = shap[j]
                if abs(value) >= THRESHOLD:  # feature appears as a component
                    component_id = 2*j if value > 0 else 2*j+1
                    components.append(component_id)
            f.write('T{};{};{}\n'.format(i, components, result))

def get_diagnosis():
    ei = readPlanningFile(r"matrix_for_SFL")
    return ei.diagnose()