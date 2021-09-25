import shap

def applySHAP(features, data, model):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data[features])

    """
    print("class 0")
    print(shap_values[0])
    print("calss 1")
    print(shap_values[1])
    """

    return shap_values
