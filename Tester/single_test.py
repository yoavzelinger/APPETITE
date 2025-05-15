from os import path as os_path
from sys import argv as sys_argv
from pandas import concat as pd_concat
from sklearn.metrics import accuracy_score
from itertools import combinations

from APPETITE import *

import Tester.TesterConstants as tester_constants
import traceback

def get_dataset(directory: str,
                file_name: str,
                proportions_tuple: int | tuple[float] = constants.PROPORTIONS_TUPLE,
                after_window_size: float = constants.AFTER_WINDOW_SIZE
                )-> Dataset:
    source = os_path.join(directory, file_name)
    return Dataset(source, proportions_tuple, after_window_size)

def get_sklearn_tree(X_train,
                     y_train):
    return build_tree(X_train, y_train)

def get_mapped_tree(sklearn_tree_model, feature_types, X_train, y_train):
    return MappedDecisionTree(sklearn_tree_model, feature_types=feature_types, X=X_train, y=y_train)

def drift_tree(mapped_tree: MappedDecisionTree,
                        dataset: Dataset,
                        ):
    """
    Generate a drifted in a multiple features
    """
    max_drift_size = min(len(mapped_tree.tree_features_set), tester_constants.MAX_DRIFT_SIZE) if tester_constants.MAX_DRIFT_SIZE > 0 else len(mapped_tree.tree_features_set)
    for after_window_test_size in tester_constants.AFTER_WINDOW_TEST_SIZES:
        print(f"\tAfter size: {after_window_test_size}%")
        dataset.after_window_size = after_window_test_size
        for drift_size in range(tester_constants.MIN_DRIFT_SIZE, max_drift_size + 1):
            print(f"\t\tDrift size: {drift_size} / {max_drift_size} features")
            for drifting_features in combinations(mapped_tree.tree_features_set, drift_size):
                print(f"\t\t\tDrifting {', '.join(drifting_features)}")
                drifted_features_types = sorted([dataset.feature_types[drifting_feature] for drifting_feature in drifting_features])
                after_drift_generator = dataset.drift_generator(drifting_features, partition="after")
                test_drift_generator = dataset.drift_generator(drifting_features, partition="test")
                for ((X_after_drifted, y_after), after_drift_description, drifted_features), ((X_test_drifted, y_test), _, _) in zip(after_drift_generator, test_drift_generator):
                    yield (X_after_drifted, y_after,), (X_test_drifted, y_test), after_drift_description[len("after") + 1: ], set(drifted_features), drifted_features_types, drift_size


def get_wasted_effort(mapped_tree: MappedDecisionTree,
                      diagnoses: list[list[int]],
                      true_faulty_features: set[str],
                      require_full_fix = True
 ) -> int:
    """
    Calculate the wasted effort of the diagnoser.
    In here the wasted effort is calculated for fixes of all the nodes that include a faulty feature.

    Parameters:
    mapped_tree (MappedDecisionTree): The mapped decision tree.
    diagnoses (list[list[int]]): The diagnoses of the nodes.
    true_faulty_features (list[str]): The true faulty features.

    Returns:
    int: The wasted effort.
    """
    # Get node's feature or it's parent's feature if it's None
    get_node_feature_func = lambda node: node.feature if node.feature is not None else node.parent.feature
    # If not require_full_fix, then the faulty feature will be counted only as one
    faulty_features_nodes = {true_faulty_feature : [] for true_faulty_feature in true_faulty_features}
    for node_index, tree_node in mapped_tree.tree_dict.items():
        tree_node_feature = get_node_feature_func(tree_node)
        if tree_node_feature in true_faulty_features:
            faulty_features_nodes[tree_node_feature].append(node_index)

    wasted_effort = 0
    all_faults_detected = lambda: not any(faulty_features_nodes.values())
    for diagnosis in diagnoses:
        for diagnosed_faulty_node in map(mapped_tree.get_node, diagnosis):
            diagnosed_faulty_feature = get_node_feature_func(diagnosed_faulty_node)
            if diagnosed_faulty_feature not in true_faulty_features: # a wasted effort
                wasted_effort += 1
            else:   # relevant fix
                if require_full_fix and diagnosed_faulty_node.sk_index in faulty_features_nodes[diagnosed_faulty_feature]:
                    faulty_features_nodes[diagnosed_faulty_feature].remove(diagnosed_faulty_node.sk_index)
                else:
                    faulty_features_nodes[diagnosed_faulty_feature] = []
        if all_faults_detected():
            break
    return wasted_effort

def get_correctly_identified_ratio(detected_faulty_features: set[str],
                                    true_faulty_features: set[str]
                                    ) -> float:
    identified_features = map(lambda feature: feature in detected_faulty_features, true_faulty_features)
    return sum(identified_features) / len(true_faulty_features)

def get_accuracy(model, X, y):
    y_predicted = model.predict(X)
    return accuracy_score(y, y_predicted)

def is_drift_contains_numeric_features(drifted_features_types):
    return any(map(lambda feature_type: feature_type == "numeric", drifted_features_types))

def is_drift_contains_binary_features(drifted_features_types):
    return any(map(lambda feature_type: feature_type == "binary", drifted_features_types))

def get_total_drift_types(drifted_features_types):
    drift_contains_numeric_features, drift_contains_binary_features = is_drift_contains_numeric_features(drifted_features_types), is_drift_contains_binary_features(drifted_features_types)
    if drift_contains_numeric_features and drift_contains_binary_features:
        return "mixed"
    if drift_contains_numeric_features:
        return "numeric"
    return "binary"

def run_single_test(directory, file_name, proportions_tuple=constants.PROPORTIONS_TUPLE, after_window_size=constants.AFTER_WINDOW_SIZE, diagnoser_names=tester_constants.constants.DEFAULT_FIXING_DIAGNOSER, *diagnoser_parameters):
    dataset = get_dataset(directory, file_name, proportions_tuple, after_window_size)

    X_train, y_train = dataset.get_before_concept()
    sklearn_tree_model = get_sklearn_tree(X_train, y_train)

    X_after, y_after = dataset.get_after_concept()
    X_test, y_test = dataset.get_test_concept()
    original_accuracy = get_accuracy(sklearn_tree_model, pd_concat([X_after, X_test]), pd_concat([y_after, y_test]))
    if original_accuracy < tester_constants.MINIMUM_ORIGINAL_ACCURACY:  # Original model is not good enough
        # print(f"Original model is not good enough, accuracy: {original_accuracy}")
        return
    
    original_after_accuracy, original_test_accuracy = get_accuracy(sklearn_tree_model, X_after, y_after), get_accuracy(sklearn_tree_model, X_test, y_test)

    mapped_tree = get_mapped_tree(sklearn_tree_model, dataset.feature_types, X_train, y_train)
    for (X_after_drifted, y_after), (X_test_drifted, y_test), drift_description, drifted_features, drifted_features_types, drift_size in drift_tree(mapped_tree, dataset):
        try:
            drifted_after_accuracy, drifted_test_accuracy = get_accuracy(mapped_tree.sklearn_tree_model, X_after_drifted, y_after), get_accuracy(mapped_tree.sklearn_tree_model, X_test_drifted, y_test)
            drifted_after_accuracy_drop, drifted_test_accuracy_drop = original_after_accuracy - drifted_after_accuracy, original_test_accuracy - drifted_test_accuracy
            if drifted_after_accuracy_drop < tester_constants.MINIMUM_DRIFT_ACCURACY_DROP or drifted_test_accuracy_drop < tester_constants.MINIMUM_DRIFT_ACCURACY_DROP:   # insignificant drift
                # print(f"Drift is insignificant, accuracy drop: after: {drifted_after_accuracy_drop}, test: {drifted_test_accuracy_drop}")
                continue

            drifted_features_types = [drifted_features_types] if isinstance(drifted_features_types, str) else drifted_features_types


            after_retrained_tree = get_sklearn_tree(X_after_drifted, y_after)
            after_retrained_accuracy = get_accuracy(after_retrained_tree, X_test_drifted, y_test)
            after_retrained_accuracy_bump = after_retrained_accuracy - drifted_test_accuracy

            X_before_after_concat, y_before_after_concat = pd_concat([X_train, X_after_drifted]), pd_concat([y_train, y_after])
            before_after_retrained_tree = get_sklearn_tree(X_before_after_concat, y_before_after_concat)
            before_after_retrained_accuracy = get_accuracy(before_after_retrained_tree, X_test_drifted, y_test)
            before_after_retrained_accuracy_bump = before_after_retrained_accuracy - drifted_test_accuracy

            current_results_dict = {
                "after size": dataset.after_proportion * dataset.after_window_size * 100,
                "drift size": drift_size,
                "drift description": drift_description,
                "drifted features types": ", ".join(drifted_features_types),
                "total drift type": get_total_drift_types(drifted_features_types),
                "tree size": mapped_tree.node_count,
                "after accuracy decrease": drifted_test_accuracy * 100,
                "after retrain accuracy": after_retrained_accuracy * 100,
                "after retrain accuracy increase": after_retrained_accuracy_bump * 100,
                "before after retrain accuracy": before_after_retrained_accuracy * 100,
                "before after retrain accuracy increase": before_after_retrained_accuracy_bump * 100
            }
            for diagnoser_name in diagnoser_names:
                fixer = Fixer(mapped_tree, X_after_drifted, y_after, diagnoser__class_name=diagnoser_name, *diagnoser_parameters)
                fixed_mapped_tree, faulty_nodes_indices = fixer.fix_tree()
                faulty_nodes = [mapped_tree.get_node(faulty_node_index) for faulty_node_index in faulty_nodes_indices]
                detected_faulty_features = set([faulty_node.feature if (faulty_node.feature or not faulty_node.is_terminal()) else "target" for faulty_node in faulty_nodes])
                fixed_test_accuracy = get_accuracy(fixed_mapped_tree.sklearn_tree_model, X_test_drifted, y_test)
                test_accuracy_bump = fixed_test_accuracy - drifted_test_accuracy
                drifted_features = drifted_features if isinstance(drifted_features, set) else set([drifted_features])
                wasted_effort = get_wasted_effort(mapped_tree, fixer.diagnoses, drifted_features, tester_constants.WASTED_EFFORT_REQUIRE_FULL_FIX)
                correctly_identified = get_correctly_identified_ratio(detected_faulty_features, drifted_features)
                current_results_dict.update({
                    f"{diagnoser_name} faulty nodes indices": ", ".join(map(str, faulty_nodes_indices)),
                    f"{diagnoser_name} faulty features": ", ".join(detected_faulty_features),
                    f"{diagnoser_name} wasted effort": wasted_effort,
                    f"{diagnoser_name} correctly_identified": correctly_identified,
                    f"{diagnoser_name} fix accuracy": fixed_test_accuracy * 100,
                    f"{diagnoser_name} fix accuracy increase": test_accuracy_bump * 100
                })
            yield current_results_dict
        except Exception as e:
            exception_class = e.__class__.__name__
            tb = traceback.extract_tb(e.__traceback__)
            error_line, error_file = tb[-1].lineno, tb[-1].filename
            diagnoser_info = f", diagnoser: {diagnoser_name}" if ('diagnoser_name' in locals() and diagnoser_name) else ""
            raise Exception(f"{exception_class} in {drift_description}, after window size: {dataset.after_window_size} ({error_file}, line {error_line}{diagnoser_info}): {e}")
        
def get_example_mapped_tree(directory=tester_constants.DATASETS_FULL_PATH, file_name=tester_constants.EXAMPLE_FILE_NAME):
    dataset = get_dataset(directory, file_name + ".csv")
    X_train, y_train = dataset.get_before_concept()
    sklearn_tree_model = get_sklearn_tree(X_train, y_train)
    return get_mapped_tree(sklearn_tree_model, dataset.feature_types, X_train, y_train)

def sanity_run(directory=tester_constants.DATASETS_FULL_PATH, file_name=tester_constants.EXAMPLE_FILE_NAME):
    for result in run_single_test(directory, file_name):
        print(result)
        
if __name__ == "__main__":
    file_name = tester_constants.EXAMPLE_FILE_NAME if len(sys_argv) < 2 else sys_argv[1]
    sanity_run(tester_constants.DATASETS_FULL_PATH, file_name + ".csv")