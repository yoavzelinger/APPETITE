from APPETITE import *
from APPETITE.Diagnosers import *
from APPETITE.Fixers import *

from sklearn.metrics import accuracy_score

from .Constants import MINIMUM_ORIGINAL_ACCURACY, MINIMUM_DRIFT_ACCURACY_DROP, DEFAULT_TESTING_DIAGNOSER

WRAP_EXCEPTION = True

def get_dataset(directory, file_name):
    return Dataset(directory + file_name)

def get_example_tree(X_train, y_train):
    return build_tree(X_train, y_train)

def get_mapped_tree(sklearn_tree_model, feature_types, X_train, y_train):
    return MappedDecisionTree(sklearn_tree_model, feature_types=feature_types, X=X_train, y=y_train)

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

def get_faulty_node(mapped_tree, X_drifted, y_original, diagnoser_name, *diagnoser_parameters):
    diagnoser_class, diagnoser_parameters = get_diagnoser(diagnoser_name, *diagnoser_parameters)
    diagnoser = diagnoser_class(mapped_tree, X_drifted, y_original, *diagnoser_parameters)
    diagnosis = diagnoser.get_diagnosis()
    return diagnosis[0]

def run_test(directory, file_name, diagnoser_names=DEFAULT_TESTING_DIAGNOSER, *diagnoser_parameters):
    dataset = get_dataset(directory, file_name)

    X_train, y_train = dataset.get_before_concept()
    sklearn_tree_model = get_example_tree(X_train, y_train)

    X_test, y_test = dataset.get_test_concept()
    no_drift_test_accuracy = get_accuracy(sklearn_tree_model, X_test, y_test)
    if no_drift_test_accuracy < MINIMUM_ORIGINAL_ACCURACY:  # Original model is not good enough
        return

    X_after_original, y_after_original = dataset.get_after_concept()
    no_drift_after_accuracy = get_accuracy(sklearn_tree_model, X_after_original, y_after_original)

    mapped_tree = get_mapped_tree(sklearn_tree_model, dataset.feature_types, X_train, y_train)

    for (X_after_drifted, y_after), (X_test_drifted, y_test), drift_description in drift_single_tree_feature(mapped_tree, dataset):
        try:
            after_accuracy = get_accuracy(mapped_tree.sklearn_tree_model, X_after_drifted, y_after) # Original model
            after_accuracy_drop = no_drift_after_accuracy - after_accuracy
            if after_accuracy_drop < MINIMUM_DRIFT_ACCURACY_DROP:   # Insigificant drift
                continue
            current_results_dict = {
                "drift description": drift_description,
                "tree size": mapped_tree.node_count,
                "after accuracy decrease precentage": after_accuracy_drop * 100
            }
            if isinstance(diagnoser_names, str):
                diagnoser_names = (diagnoser_names, )
            for diagnoser_name in diagnoser_names:
                faulty_node_index = get_faulty_node(mapped_tree, X_after_drifted, y_after, diagnoser_name, *diagnoser_parameters)
                faulty_feature = mapped_tree.get_node(faulty_node_index).feature
                fixer = Fixer(mapped_tree, X_after_drifted, y_after, diagnoser_name=diagnoser_name, *diagnoser_parameters)
                fixed_mapped_tree = fixer.fix_single_fault()
                test_accuracy = get_accuracy(mapped_tree.sklearn_tree_model, X_test_drifted, y_test) # Original model
                fixed_test_accuracy = get_accuracy(fixed_mapped_tree.sklearn_tree_model, X_test_drifted, y_test)
                test_accuracy_bump = fixed_test_accuracy - test_accuracy
                current_results_dict.update({
                    f"{diagnoser_name} faulty node index": faulty_node_index,
                    f"{diagnoser_name} faulty feature": faulty_feature,
                    f"{diagnoser_name} fix accuracy increase precentage": test_accuracy_bump * 100
                })
            yield current_results_dict
        except Exception as e:
            if WRAP_EXCEPTION:
                raise Exception(f"Error in {drift_description}: {e}")
            e.add_note(f"Error in {drift_description}")
            raise e

DIRECTORY = "data\\Classification_Datasets\\"
EXAMPLE_FILE_NAME = "Acute-Inflammation"
EXAMPLE_FILE_NAME = "Acute-Inflammation" + ".csv"
def get_example_mapped_tree(directory=DIRECTORY, file_name=EXAMPLE_FILE_NAME):
    dataset = get_dataset(directory, file_name)
    X_train, y_train = dataset.get_before_concept()
    sklearn_tree_model = get_example_tree(X_train, y_train)
    return get_mapped_tree(sklearn_tree_model, dataset.feature_types, X_train, y_train)

def sanity_run(directory=DIRECTORY, file_name=EXAMPLE_FILE_NAME):
    global WRAP_EXCEPTION
    WRAP_EXCEPTION = False
    for result in run_test(directory, file_name):
        print(result)
    WRAP_EXCEPTION = True
        
if __name__ == "__main__":
    sanity_run(DIRECTORY, EXAMPLE_FILE_NAME)