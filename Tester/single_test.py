import os
import sys
import pandas as pd
from itertools import combinations

import traceback

from APPETITE import *

import Tester.TesterConstants as tester_constants
from Tester.metrics import get_accuracy, get_wasted_effort, get_correctly_identified_ratio

def get_dataset(directory: str,
                file_name: str,
                file_extension: str = ".csv",
                proportions_tuple: int | tuple[float] = constants.PROPORTIONS_TUPLE,
                after_window_size: float = constants.AFTER_WINDOW_SIZE
                )-> Dataset:
    source = os.path.join(directory, f"{file_name}{file_extension}")
    return Dataset(source, proportions_tuple, after_window_size)

def get_sklearn_tree(X_train,
                     y_train,
                     is_retraining_model: bool = False):
    return build_tree(X_train, y_train, is_retraining_model=is_retraining_model)

def get_mapped_tree(sklearn_tree_model, feature_types, X_train, y_train):
    return MappedDecisionTree(sklearn_tree_model, feature_types=feature_types, X=X_train, y=y_train)

def drift_tree(mapped_tree: MappedDecisionTree,
               dataset: Dataset,
               after_window_test_sizes: list[float] = tester_constants.AFTER_WINDOW_TEST_SIZES,
               min_drift_size: int = tester_constants.MIN_DRIFT_SIZE,
               max_drift_size: int = tester_constants.MAX_DRIFT_SIZE
               ):
    """
    Generate a drifted in a multiple features
    """
    current_min_drift_size = max(min_drift_size, 1)
    current_max_drift_size = len(mapped_tree.tree_features_set)
    if max_drift_size > 0:
        current_max_drift_size = min(current_max_drift_size, max_drift_size)
    for after_window_test_size in after_window_test_sizes:
        print(f"\tAfter size: {after_window_test_size}%")
        dataset.update_after_window_size(after_window_test_size)
        for drift_size in range(current_min_drift_size, current_max_drift_size + 1):
            print(f"\t\tDrift size: {drift_size} / {current_max_drift_size} features")
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

def run_single_test(directory, file_name, file_extension: str = ".csv", proportions_tuple=constants.PROPORTIONS_TUPLE, after_window_test_sizes=tester_constants.AFTER_WINDOW_TEST_SIZES, min_drift_size=tester_constants.MIN_DRIFT_SIZE, max_drift_size=tester_constants.MAX_DRIFT_SIZE, diagnosers_data=tester_constants.DEFAULT_TESTING_DIAGNOSER):
    dataset = get_dataset(directory, file_name, file_extension=file_extension, proportions_tuple=proportions_tuple)

    X_train, y_train = dataset.get_before_concept()
    sklearn_tree_model = get_sklearn_tree(X_train, y_train)

    X_after, y_after = dataset.get_after_concept()
    X_test, y_test = dataset.get_test_concept()
    original_accuracy = get_accuracy(sklearn_tree_model, pd.concat([X_after, X_test]), pd.concat([y_after, y_test]))
    if original_accuracy < tester_constants.MINIMUM_ORIGINAL_ACCURACY:  # Original model is not good enough
        # print(f"Original model is not good enough, accuracy: {original_accuracy}")
        return
    
    original_after_accuracy, original_test_accuracy = get_accuracy(sklearn_tree_model, X_after, y_after), get_accuracy(sklearn_tree_model, X_test, y_test)

    mapped_tree = get_mapped_tree(sklearn_tree_model, dataset.feature_types, X_train, y_train)
    for (X_after_drifted, y_after), (X_test_drifted, y_test), (drift_severity_level, drift_description), drifted_features, drifted_features_types, drift_size in drift_tree(mapped_tree, dataset, after_window_test_sizes=after_window_test_sizes, min_drift_size=min_drift_size, max_drift_size=max_drift_size):
        try:
            drifted_after_accuracy, drifted_test_accuracy = get_accuracy(mapped_tree.sklearn_tree_model, X_after_drifted, y_after), get_accuracy(mapped_tree.sklearn_tree_model, X_test_drifted, y_test)
            drifted_after_accuracy_drop, drifted_test_accuracy_drop = original_after_accuracy - drifted_after_accuracy, original_test_accuracy - drifted_test_accuracy
            if drifted_after_accuracy_drop < tester_constants.MINIMUM_DRIFT_ACCURACY_DROP or drifted_test_accuracy_drop < tester_constants.MINIMUM_DRIFT_ACCURACY_DROP:   # insignificant drift
                # print(f"Drift is insignificant, accuracy drop: after: {drifted_after_accuracy_drop}, test: {drifted_test_accuracy_drop}")
                continue

            print(f"\t\t\t\tDiagnosing")

            drifted_features_types = [drifted_features_types] if isinstance(drifted_features_types, str) else drifted_features_types


            X_before_after_concat, y_before_after_concat = pd.concat([X_train, X_after_drifted]), pd.concat([y_train, y_after])
            before_after_retrained_tree = get_sklearn_tree(X_before_after_concat, y_before_after_concat, is_retraining_model=True)
            before_after_retrained_accuracy = get_accuracy(before_after_retrained_tree, X_test_drifted, y_test)
            before_after_retrained_accuracy_bump = before_after_retrained_accuracy - drifted_test_accuracy

            after_retrained_tree = get_sklearn_tree(X_after_drifted, y_after, is_retraining_model=True)
            after_retrained_accuracy = get_accuracy(after_retrained_tree, X_test_drifted, y_test)
            after_retrained_accuracy_bump = after_retrained_accuracy - drifted_test_accuracy

            drifted_features = drifted_features if isinstance(drifted_features, set) else set([drifted_features])
            faulty_features_nodes = get_drifted_nodes(mapped_tree, drifted_features)

            current_results_dict = {
                tester_constants.DATASET_COLUMN_NAME: file_name,
                tester_constants.TREE_SIZE_COLUMN_NAME: mapped_tree.node_count,
                tester_constants.TREE_FEATURES_COUNT_COLUMN_NAME: len(mapped_tree.tree_features_set),
                tester_constants.AFTER_SIZE_COLUMN_NAME: dataset.after_proportion * dataset.after_window_size * 100,
                tester_constants.DRIFT_SIZE_COLUMN_NAME: drift_size,
                tester_constants.TOTAL_DRIFT_TYPE_COLUMN_NAME: get_total_drift_types(drifted_features_types),
                tester_constants.DRIFT_SEVERITY_LEVEL_COLUMN_NAME: drift_severity_level,
                tester_constants.DRIFTED_FEATURES_COLUMN_NAME: ", ".join(map(lambda feature: f"{feature}: {faulty_features_nodes[feature]}", faulty_features_nodes)),
                tester_constants.DRIFTED_FEATURES_TYPES_COLUMN_NAME: ", ".join(drifted_features_types),
                tester_constants.DRIFT_DESCRIPTION_COLUMN_NAME: drift_description,
                tester_constants.AFTER_ACCURACY_DECREASE_COLUMN_NAME: drifted_test_accuracy * 100,
                f"{tester_constants.AFTER_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_NAME_SUFFIX}": after_retrained_accuracy * 100,
                f"{tester_constants.AFTER_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_INCREASE_NAME_SUFFIX}": after_retrained_accuracy_bump * 100,
                f"{tester_constants.BEFORE_AFTER_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_NAME_SUFFIX}": before_after_retrained_accuracy * 100,
                f"{tester_constants.BEFORE_AFTER_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_INCREASE_NAME_SUFFIX}": before_after_retrained_accuracy_bump * 100
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
                    f"{diagnoser_output_name} {tester_constants.FAULTY_FEATURES_NAME_SUFFIX}": ", ".join(detected_faulty_features),
                    f"{diagnoser_output_name} {tester_constants.DIAGNOSES_NAME_SUFFIX}": ", ".join(map(str, diagnoses)),
                    f"{diagnoser_output_name} {tester_constants.WASTED_EFFORT_NAME_SUFFIX}": wasted_effort,
                    f"{diagnoser_output_name} {tester_constants.CORRECTLY_IDENTIFIED_NAME_SUFFIX}": correctly_identified * 100,
                    f"{diagnoser_output_name} {tester_constants.FIX_ACCURACY_NAME_SUFFIX}": fixed_test_accuracy * 100,
                    f"{diagnoser_output_name} {tester_constants.FIX_ACCURACY_INCREASE_NAME_SUFFIX}": test_accuracy_bump * 100
                })
            yield current_results_dict
        except Exception as e:
            exception_class = e.__class__.__name__
            diagnoser_info = f"(diagnoser: {diagnoser_class_name} - {diagnoser_output_name})" if 'diagnoser_output_name' in locals() else ""
            yield Exception(f"{exception_class} in {drift_description}, after window size: {dataset.after_window_size} {diagnoser_info} :\n "
                            f"{''.join(traceback.format_exception(type(e), e, e.__traceback__))}")
        
def get_example_mapped_tree(directory=tester_constants.DATASETS_DIRECTORY_FULL_PATH, file_name=tester_constants.EXAMPLE_FILE_NAME):
    dataset = get_dataset(directory, file_name)
    X_train, y_train = dataset.get_before_concept()
    sklearn_tree_model = get_sklearn_tree(X_train, y_train)
    return get_mapped_tree(sklearn_tree_model, dataset.feature_types, X_train, y_train)

def sanity_run(directory=tester_constants.DATASETS_DIRECTORY_FULL_PATH, file_name=tester_constants.EXAMPLE_FILE_NAME, diagnosers_data=tester_constants.DEFAULT_TESTING_DIAGNOSER):
    for result in run_single_test(directory=directory, file_name=file_name, diagnosers_data=diagnosers_data):
        print(result)
        
if __name__ == "__main__":
    file_name = tester_constants.EXAMPLE_FILE_NAME if len(sys.argv) < 2 else sys.argv[1]
    sanity_run(tester_constants.DATASETS_DIRECTORY_FULL_PATH, file_name)