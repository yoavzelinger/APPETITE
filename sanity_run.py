from APPETITE import *
from APPETITE.Diagnosers import *
from APPETITE.Fixers import *

from sklearn.metrics import accuracy_score

DIRECTORY = "data\\Classification_Datasets\\"
FILE_NAME = "abalone.csv"

def get_dataset():
    return Dataset(DIRECTORY + FILE_NAME)

def get_example_tree():
    return build_tree(X_train, y_train)

def get_mapped_tree():
    return MappedDecisionTree(sklearn_tree_model, feature_types=feature_types)

def drift_single_tree_feature():
    tree_features_set = mapped_tree.tree_features_set
    for drifting_feature in tree_features_set:
        for (X_after_drifted, y_after), drift_description in dataset.drift_generator(drifting_feature):
            yield (X_after_drifted, y_after), drift_description

def print_single_tree_feature_drift_accuracy():
    for (X_after_drifted, y_after), drift_description in drift_single_tree_feature():
        y_after__drifted_predicted = sklearn_tree_model.predict(X_after_drifted)
        print(f"After drift {drift_description} accuracy: ", accuracy_score(y_after, y_after__drifted_predicted))

def faulty_node_index_generator():
    for (X_after_drifted, y_after), drift_description in drift_single_tree_feature():
        diagnoser = SFLDT(mapped_tree, X_after_drifted, y_after)
        diagnosis = diagnoser.get_diagnosis()
        yield drift_description, diagnosis[0]

dataset = get_dataset()
X_train, y_train = dataset.get_before_concept()

sklearn_tree_model = get_example_tree()

X_after_original, y_after_original = dataset.get_after_concept()
y_after_original_predicted = sklearn_tree_model.predict(X_after_original)
print("No drift accuracy: ", accuracy_score(y_after_original, y_after_original_predicted))

feature_types = dataset.feature_types
mapped_tree = get_mapped_tree()

print_single_tree_feature_drift_accuracy()

for drift_description, faulty_node_index in faulty_node_index_generator():
    faulty_feature = mapped_tree.get_node(faulty_node_index).feature
    print(f"Drift {drift_description} detected faulty node in index {faulty_node_index}. The node's feature is {faulty_feature}.")