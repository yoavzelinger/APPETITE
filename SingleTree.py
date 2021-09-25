from DataSet import DataSet
from SHAP import applySHAP
from buildModel import build_model
from sklearn import metrics
from SFL import build_SFL_matrix

dataset = DataSet("data/mixed_0101_abrupto.csv", "abrupt", ["X1", "X2", "X3", "X4"], "class")
SIZE = 10000

if __name__ == '__main__':
    model = build_model(dataset.data.iloc[0:SIZE], ["X1", "X2", "X3", "X4"], "class")

    # check model accuracy after concept drift
    new_data = dataset.data.iloc[SIZE: int(1.1*SIZE)]
    new_data_x = new_data[dataset.features]
    y_pred = model.predict(new_data_x)
    new_data_y = new_data[dataset.target]
    print("Accuracy after concept drift:", metrics.accuracy_score(new_data_y, y_pred))

    # update model
    shap_values = applySHAP(dataset.features, new_data, model)
    build_SFL_matrix(dataset.features, shap_values, y_pred, new_data_y)





