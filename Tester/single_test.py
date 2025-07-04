from os import path as os_path
from sys import argv as sys_argv
from pandas import concat as pd_concat
from itertools import combinations

from APPETITE import *

import Tester.TesterConstants as tester_constants
from Tester.metrics import get_accuracy, get_wasted_effort, get_correctly_identified_ratio
import traceback

def get_dataset(directory: str,
                file_name: str,
                proportions_tuple: int | tuple[float] = constants.PROPORTIONS_TUPLE,
                after_window_size: float = constants.AFTER_WINDOW_SIZE
                )-> Dataset:
    source = os_path.join(directory, file_name)
    return Dataset(source, proportions_tuple, after_window_size)

def get_sklearn_tree(X_train,
                     y_train,
                     is_retraining_model: bool = False):
    return build_tree(X_train, y_train, is_retraining_model=is_retraining_model)

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
        dataset.update_after_window_size(after_window_test_size)
        for drift_size in range(tester_constants.MIN_DRIFT_SIZE, max_drift_size + 1):
            print(f"\t\tDrift size: {drift_size} / {max_drift_size} features")
            for drifting_features in combinations(mapped_tree.tree_features_set, drift_size):
                print(f"\t\t\tDrifting {', '.join(drifting_features)}")
                drifted_features_types = sorted([dataset.feature_types[drifting_feature] for drifting_feature in drifting_features])
                for (X_after_drifted, y_after), (X_test_drifted, y_test), drift_severity_level, drift_description in dataset.drift_generator(drifting_features, partition="after"):
                    yield (X_after_drifted, y_after), (X_test_drifted, y_test), (drift_severity_level, drift_description), set(drifting_features), drifted_features_types, drift_size

def get_drifted_nodes(mapped_tree: MappedDecisionTree,
                      drifted_features: set[str]
 ) -> dict[str, list[int]]:
    faulty_features_nodes = {true_faulty_feature : [] for true_faulty_feature in drifted_features}
    for node_index, tree_node in mapped_tree.tree_dict.items():
        tree_node_feature = tree_node.feature
        if tree_node_feature in drifted_features:
            faulty_features_nodes[tree_node_feature].append(node_index)
    return faulty_features_nodes

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

def run_single_test(directory, file_name, proportions_tuple=constants.PROPORTIONS_TUPLE, after_window_size=constants.AFTER_WINDOW_SIZE, diagnosers_data=tester_constants.DEFAULT_TESTING_DIAGNOSER):
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
    for (X_after_drifted, y_after), (X_test_drifted, y_test), (drift_severity_level, drift_description), drifted_features, drifted_features_types, drift_size in drift_tree(mapped_tree, dataset):
        try:
            drifted_after_accuracy, drifted_test_accuracy = get_accuracy(mapped_tree.sklearn_tree_model, X_after_drifted, y_after), get_accuracy(mapped_tree.sklearn_tree_model, X_test_drifted, y_test)
            drifted_after_accuracy_drop, drifted_test_accuracy_drop = original_after_accuracy - drifted_after_accuracy, original_test_accuracy - drifted_test_accuracy
            if drifted_after_accuracy_drop < tester_constants.MINIMUM_DRIFT_ACCURACY_DROP or drifted_test_accuracy_drop < tester_constants.MINIMUM_DRIFT_ACCURACY_DROP:   # insignificant drift
                # print(f"Drift is insignificant, accuracy drop: after: {drifted_after_accuracy_drop}, test: {drifted_test_accuracy_drop}")
                continue

            print(f"\t\t\t\tDiagnosing")

            drifted_features_types = [drifted_features_types] if isinstance(drifted_features_types, str) else drifted_features_types


            X_before_after_concat, y_before_after_concat = pd_concat([X_train, X_after_drifted]), pd_concat([y_train, y_after])
            before_after_retrained_tree = get_sklearn_tree(X_before_after_concat, y_before_after_concat, is_retraining_model=True)
            before_after_retrained_accuracy = get_accuracy(before_after_retrained_tree, X_test_drifted, y_test)
            before_after_retrained_accuracy_bump = before_after_retrained_accuracy - drifted_test_accuracy

            after_retrained_tree = get_sklearn_tree(X_after_drifted, y_after, is_retraining_model=True)
            after_retrained_accuracy = get_accuracy(after_retrained_tree, X_test_drifted, y_test)
            after_retrained_accuracy_bump = after_retrained_accuracy - drifted_test_accuracy

            drifted_features = drifted_features if isinstance(drifted_features, set) else set([drifted_features])
            faulty_features_nodes = get_drifted_nodes(mapped_tree, drifted_features)

            current_results_dict = {
                "after size": dataset.after_proportion * dataset.after_window_size * 100,
                "drift size": drift_size,
                "drift severity level": drift_severity_level,
                "drift description": drift_description,
                "drifted features": ", ".join(map(lambda feature: f"{feature}: {faulty_features_nodes[feature]}", faulty_features_nodes)),
                "drifted features types": ", ".join(drifted_features_types),
                "total drift type": get_total_drift_types(drifted_features_types),
                "tree size": mapped_tree.node_count,
                "after accuracy decrease": drifted_test_accuracy * 100,
                "after retrain accuracy": after_retrained_accuracy * 100,
                "after retrain accuracy increase": after_retrained_accuracy_bump * 100,
                "before after retrain accuracy": before_after_retrained_accuracy * 100,
                "before after retrain accuracy increase": before_after_retrained_accuracy_bump * 100
            }

            for diagnoser_data in diagnosers_data:
                diagnoser_output_name, diagnoser_class_name, diagnoser_parameters = diagnoser_data["output_name"], diagnoser_data["class_name"], diagnoser_data["parameters"]
                fixer = Fixer(mapped_tree, X_after_drifted, y_after, diagnoser__class_name=diagnoser_class_name, diagnoser_parameters=diagnoser_parameters, diagnoser_output_name=diagnoser_output_name)
                fixed_mapped_tree, faulty_nodes_indices = fixer.fix_tree()
                faulty_nodes = [mapped_tree.get_node(faulty_node_index) for faulty_node_index in faulty_nodes_indices]
                detected_faulty_features = set([faulty_node.feature if not faulty_node.is_terminal() else "target" for faulty_node in faulty_nodes])
                fixed_test_accuracy = get_accuracy(fixed_mapped_tree.sklearn_tree_model, X_test_drifted, y_test)
                test_accuracy_bump = fixed_test_accuracy - drifted_test_accuracy
                diagnoses = fixer.diagnoses
                wasted_effort = get_wasted_effort(mapped_tree, diagnoses, faulty_features_nodes)
                correctly_identified = get_correctly_identified_ratio(detected_faulty_features, drifted_features)
                current_results_dict.update({
                    f"{diagnoser_output_name} faulty features": ", ".join(detected_faulty_features),
                    f"{diagnoser_output_name} diagnoses": ", ".join(map(str, diagnoses)),
                    f"{diagnoser_output_name} wasted effort": wasted_effort,
                    f"{diagnoser_output_name} correctly_identified": correctly_identified,
                    f"{diagnoser_output_name} fix accuracy": fixed_test_accuracy * 100,
                    f"{diagnoser_output_name} fix accuracy increase": test_accuracy_bump * 100
                })
            yield current_results_dict
        except Exception as e:
            exception_class = e.__class__.__name__
            tb = traceback.extract_tb(e.__traceback__)
            error_line, error_file = tb[-1].lineno, tb[-1].filename
            diagnoser_info = f", diagnoser: {diagnoser_output_name} ({diagnoser_class_name})" if ('diagnoser_output_name' in locals() and diagnoser_output_name) else ""
            yield Exception(f"{exception_class} in {drift_description}, after window size: {dataset.after_window_size} ({error_file}, line {error_line}{diagnoser_info}: {e}")
        
def get_example_mapped_tree(directory=tester_constants.DATASETS_FULL_PATH, file_name=tester_constants.EXAMPLE_FILE_NAME):
    dataset = get_dataset(directory, file_name + ".csv")
    X_train, y_train = dataset.get_before_concept()
    sklearn_tree_model = get_sklearn_tree(X_train, y_train)
    return get_mapped_tree(sklearn_tree_model, dataset.feature_types, X_train, y_train)

def sanity_run(directory=tester_constants.DATASETS_FULL_PATH, file_name=tester_constants.EXAMPLE_FILE_NAME, diagnosers_data=tester_constants.DEFAULT_TESTING_DIAGNOSER):
    for result in run_single_test(directory=directory, file_name=file_name + ".csv", diagnosers_data=diagnosers_data):
        print(result)
        
if __name__ == "__main__":
    file_name = tester_constants.EXAMPLE_FILE_NAME if len(sys_argv) < 2 else sys_argv[1]
    sanity_run(tester_constants.DATASETS_FULL_PATH, file_name + ".csv")