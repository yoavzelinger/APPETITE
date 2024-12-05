from APPETITE import *
from APPETITE.Diagnosers import *
from APPETITE.Fixers import *

from sklearn.metrics import accuracy_score

from .Constants import MINIMUM_ORIGINAL_ACCURACY, MINIMUM_DRIFT_ACCURACY_DROP

def get_dataset(directory, file_name):
    return Dataset(directory + file_name)

def get_example_tree(X_train, y_train):
    return build_tree(X_train, y_train)

def get_mapped_tree(sklearn_tree_model, feature_types, X_train):
    return MappedDecisionTree(sklearn_tree_model, feature_types=feature_types, data=X_train)

def drift_single_tree_feature(mapped_tree, dataset):
    """
    Generate a drifted in a single tree that is used in the tree structure.
    """
    tree_features_set = mapped_tree.tree_features_set
    for drifting_feature in tree_features_set:
        after_drift_generator = dataset.drift_generator(drifting_feature, partition="after")
        test_drift_generator = dataset.drift_generator(drifting_feature, partition="test")
        for ((X_after_drifted, y_after), after_drift_description), ((X_test_drifted, y_test), _) in zip(after_drift_generator, test_drift_generator):
            yield (X_after_drifted, y_after), (X_test_drifted, y_test), after_drift_description[6: ]

def get_accuracy(model, X, y):
    y_predicted = model.predict(X)
    return accuracy_score(y, y_predicted)

def get_faulty_node(mapped_tree, X_drifted, y_original):
    diagnoser = SFLDT(mapped_tree, X_drifted, y_original)
    diagnosis = diagnoser.get_diagnosis()
    return diagnosis[0]

def run_test(directory, file_name):
    dataset = get_dataset(directory, file_name)

    X_train, y_train = dataset.get_before_concept()
    sklearn_tree_model = get_example_tree(X_train, y_train)

    X_test, y_test = dataset.get_test_concept()
    no_drift_test_accuracy = get_accuracy(sklearn_tree_model, X_test, y_test)
    if no_drift_test_accuracy < MINIMUM_ORIGINAL_ACCURACY:  # Original model is not good enough
        return

    X_after_original, y_after_original = dataset.get_after_concept()
    no_drift_after_accuracy = get_accuracy(sklearn_tree_model, X_after_original, y_after_original)

    mapped_tree = get_mapped_tree(sklearn_tree_model, dataset.feature_types, X_train)

    for (X_after_drifted, y_after), (X_test_drifted, y_test), drift_description in drift_single_tree_feature(mapped_tree, dataset):
        try:
            after_accuracy = get_accuracy(mapped_tree.sklearn_tree_model, X_after_drifted, y_after) # Original model
            after_accuracy_drop = no_drift_after_accuracy - after_accuracy
            if after_accuracy_drop < MINIMUM_DRIFT_ACCURACY_DROP:   # Insigificant drift
                continue
            faulty_node_index = get_faulty_node(mapped_tree, X_after_drifted, y_after)
            faulty_feature = mapped_tree.get_node(faulty_node_index).feature
            fixer = Fixer(mapped_tree, X_after_drifted, y_after)
            fixed_mapped_tree = fixer.fix_single_fault()
            fixed_mapped_tree.update_tree_attributes(X_train)
            test_accuracy = get_accuracy(mapped_tree.sklearn_tree_model, X_test_drifted, y_test) # Original model
            fixed_test_accuracy = get_accuracy(fixed_mapped_tree.sklearn_tree_model, X_test_drifted, y_test)
            test_accuracy_bump = fixed_test_accuracy - test_accuracy
            yield {
                "drift description": drift_description,
                "tree size": mapped_tree.node_count,
                "after accuracy decrease precentage": after_accuracy_drop * 100,
                "faulty node index": faulty_node_index,
                "faulty feature": faulty_feature,
                "fix accuracy increase precentage": test_accuracy_bump * 100
            }
        except Exception as e:
            raise Exception(f"Error in {drift_description}: {e}")

DIRECTORY = "data\\Classification_Datasets\\"
def get_example_mapped_tree(file_name):
    dataset = get_dataset(DIRECTORY, file_name + ".csv")
    X_train, y_train = dataset.get_before_concept()
    sklearn_tree_model = get_example_tree(X_train, y_train)
    return get_mapped_tree(sklearn_tree_model, dataset.feature_types, X_train)

def sanity_run(directory, file_name):
    for result in run_test(directory, file_name):
        print(result)
        
EXAMPLE_FILE_NAME = "Acute-Inflammation.csv"
if __name__ == "__main__":
    sanity_run(DIRECTORY, EXAMPLE_FILE_NAME)