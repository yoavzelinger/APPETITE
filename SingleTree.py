from DataSet import DataSet
from SHAP import applySHAP
from buildModel import build_model
from sklearn import metrics
from SFL import build_SFL_matrix, get_diagnosis
from updateModel import tree_to_code, print_tree_rules

dataset = DataSet("data/stagger_2102_abrupto.csv", "abrupt", ["X1", "X2", "X3"], "class")
SIZE = 10000

if __name__ == '__main__':
    model = build_model(dataset.data.iloc[0:SIZE], dataset.features, dataset.target)

    # check model accuracy after concept drift
    new_data = dataset.data.iloc[SIZE: int(1.1*SIZE)]
    new_data_x = new_data[dataset.features]
    prediction = model.predict(new_data_x)
    new_data_y = new_data[dataset.target]
    accuracy = metrics.accuracy_score(new_data_y, prediction)
    print("Accuracy of original model on data after concept drift:", accuracy)

    # update model
    shap_values = applySHAP(dataset.features, new_data, model)
    build_SFL_matrix(dataset.features, shap_values, prediction, new_data_y, dataset.name)
    diagnosis = get_diagnosis()
    print(diagnosis)
    # TODO: rate diagnosis
    # TODO: fix model

    #tree_to_code(model, dataset.features)
    #print_tree_rules(model, dataset.features)


    # TEST
    test_set = dataset.data.iloc[int(1.1 * SIZE): int(1.2 * SIZE)]
    test_set_x = test_set[dataset.features]
    test_set_y = test_set[dataset.target]

    prediction1 = model.predict(test_set_x)
    accuracy = metrics.accuracy_score(test_set_y, prediction1)
    print("Accuracy of ORIGINAL model:", accuracy)

    model_all = build_model(dataset.data.iloc[0:int(1.1 * SIZE)], dataset.features, dataset.target)
    prediction2 = model_all.predict(test_set_x)
    accuracy = metrics.accuracy_score(test_set_y, prediction2)
    print("Accuracy of a model trained with some data after concept drift:", accuracy)







