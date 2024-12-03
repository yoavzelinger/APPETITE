from DataManagementTools.Dataset import Dataset
from DecisionTreeTools.MappedDecisionTree import MappedDecisionTree
from DecisionTreeTools.DecisionTreeClassifierBuilder import build as build_tree 
from Diagnosers.SFLDT import SFLDT

from sklearn.metrics import accuracy_score

DIRECTORY = "data\\real\\"
FILE_NAME = "iris.data"

def get_dataset():
    return Dataset(DIRECTORY + FILE_NAME)

def get_example_tree():
    return build_tree(X_train, y_train)

def get_mapped_tree():
    return MappedDecisionTree(sklearn_tree_model, feature_types=feature_types)

dataset = get_dataset()
X_train, y_train = dataset.get_before_concept()
feature_types = dataset.feature_types
sklearn_tree_model = get_example_tree()
X_after, y_after = dataset.get_after_concept()
y_after_predicted = sklearn_tree_model.predict(X_after)
print("No drift accuracy: ", accuracy_score(y_after, y_after_predicted))

drifted_feature = list(feature_types.keys())[0]
(X_after, y_after), drift_description = dataset.get_feature_first_drift(drifted_feature)
y_after_predicted = sklearn_tree_model.predict(X_after)
print(f"After drift {drifted_feature} accuracy: ", accuracy_score(y_after, y_after_predicted))

mapped_tree = get_mapped_tree()

diagnoser = SFLDT(mapped_tree, X_after, y_after)
print(diagnoser.get_diagnosis())
