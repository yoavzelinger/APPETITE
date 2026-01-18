import os
import sys
import pandas as pd
from itertools import combinations
from copy import deepcopy

import traceback

from sklearn.base import ClassifierMixin

from APPETITE import *

import Tester.TesterConstants as tester_constants
from Tester.DataManagementTools import *
from Tester.DecisionTreeClassifierBuilder import build_tree
from Tester.metrics import get_accuracy, get_wasted_effort, get_correctly_identified_ratio

def get_dataset(directory: str,
                file_name: str,
                file_extension: str = ".csv"
                )-> Dataset:
    source = os.path.join(directory, f"{file_name}{file_extension}")
    return Dataset(source)

def get_sklearn_model(X_train,
                      y_train,
                      previous_model = None
 ) -> ClassifierMixin:
    return build_tree(X_train, y_train, model=deepcopy(previous_model))

def get_mapped_model(sklearn_model, feature_types, X_train, y_train):
    return MappedDecisionTree(sklearn_model, feature_types=feature_types, X=X_train, y=y_train)

def drift_tree(mapped_model: ATreeBasedMappedModel,
               dataset: Dataset,
               repair_window_test_sizes: list[float] = tester_constants.REPAIR_WINDOW_TEST_SIZES,
               min_drift_size: int = tester_constants.MIN_DRIFT_SIZE,
               max_drift_size: int = tester_constants.MAX_DRIFT_SIZE
               ):
    """
    Generate a drifted in a multiple features
    """
    current_min_drift_size = max(min_drift_size, 1)
    current_max_drift_size = len(mapped_model.model_used_features)
    if max_drift_size > 0:
        current_max_drift_size = min(current_max_drift_size, max_drift_size)
    for repair_window_test_size in repair_window_test_sizes:
        print(f"\tRepair window size: {repair_window_test_size}%")
        dataset.update_repair_window_size(repair_window_test_size)
        for drift_size in range(current_min_drift_size, current_max_drift_size + 1):
            print(f"\t\tDrift size: {drift_size} / {current_max_drift_size} features")
            for drifting_features in combinations(mapped_model.model_used_features, drift_size):
                print(f"\t\t\tDrifting {', '.join(drifting_features)}")
                drifted_features_types = sorted([dataset.feature_types[drifting_feature] for drifting_feature in drifting_features])
                for (X_repair, y_repair), (X_test, y_test), drift_severity_level, drift_description in dataset.drift_generator(drifting_features, partition="after"):
                    yield (X_repair, y_repair), (X_test, y_test), (drift_severity_level, drift_description), set(drifting_features), drifted_features_types, drift_size

def get_drifted_nodes(mapped_model: ATreeBasedMappedModel,
                      drifted_features: set[str]
 ) -> dict[str, list[int]]:
    faulty_features_nodes = {true_faulty_feature : [] for true_faulty_feature in drifted_features}
    for node_index, node in mapped_model.components_map.items():
        if node.feature in drifted_features:
            faulty_features_nodes[node.feature].append(node_index)
    return faulty_features_nodes

def is_drift_contains_numeric_features(drifted_features_types):
    return any(map(lambda feature_type: feature_type == tester_constants.FeatureType.Numeric, drifted_features_types))

def is_drift_contains_binary_features(drifted_features_types):
    return any(map(lambda feature_type: feature_type == tester_constants.FeatureType.Binary, drifted_features_types))

def get_total_drift_types(drifted_features_types: list[tester_constants.FeatureType]) -> tester_constants.FeatureType:
    drift_contains_numeric_features, drift_contains_binary_features = is_drift_contains_numeric_features(drifted_features_types), is_drift_contains_binary_features(drifted_features_types)
    if drift_contains_numeric_features and drift_contains_binary_features:
        return tester_constants.FeatureType.Mixed.name
    if drift_contains_numeric_features:
        return tester_constants.FeatureType.Numeric.name
    return tester_constants.FeatureType.Binary.name

def get_drifted_features_types_string(drifted_features_types: list[tester_constants.FeatureType]) -> str:
    return ", ".join(map(lambda feature_type: feature_type.name, drifted_features_types))

def run_single_test(directory, file_name, file_extension: str = ".csv", repair_window_test_sizes=tester_constants.REPAIR_WINDOW_TEST_SIZES, min_drift_size=tester_constants.MIN_DRIFT_SIZE, max_drift_size=tester_constants.MAX_DRIFT_SIZE, diagnosers_data: list[dict[str, object]] = tester_constants.DEFAULT_TESTING_DIAGNOSER, fixers_data: list[dict[str, object]] = tester_constants.DEFAULT_TESTING_FIXER):
    dataset = get_dataset(directory, file_name, file_extension=file_extension)

    X_before, y_before = dataset.get_before_concept_data()
    
    sklearn_model = get_sklearn_model(X_before, y_before)
    pre_drift_accuracy = get_accuracy(sklearn_model, *dataset.get_test_data())

    if pre_drift_accuracy < tester_constants.MINIMUM_ORIGINAL_ACCURACY:
        # Original model is not good enough
        return
    
    mapped_model = get_mapped_model(sklearn_model, dataset.feature_types, X_before, y_before)
    
    pre_drift_repair_accuracy = get_accuracy(sklearn_model, *dataset.get_repair_data())
    
    for (X_repair, y_repair), (X_test, y_test), (drift_severity_level, drift_description), drifted_features, drifted_features_types, drift_size in drift_tree(mapped_model, dataset, repair_window_test_sizes=repair_window_test_sizes, min_drift_size=min_drift_size, max_drift_size=max_drift_size):
        try:
            if X_repair.empty or X_test.empty:
                continue
            
            post_drift_test_accuracy = get_accuracy(mapped_model.model, X_test, y_test)
            post_drift_test_accuracy_drop = pre_drift_accuracy - post_drift_test_accuracy
            
            if pre_drift_repair_accuracy - get_accuracy(sklearn_model, X_repair, y_repair) < tester_constants.MINIMUM_DRIFT_ACCURACY_DROP or post_drift_test_accuracy_drop < tester_constants.MINIMUM_DRIFT_ACCURACY_DROP:   # insignificant drift
                continue

            faulty_features_nodes = get_drifted_nodes(mapped_model, drifted_features)

            print(f"\t\t\t\tDiagnosing")

            X_before_repair, y_before_repair = pd.concat([X_before, X_repair]).reset_index(drop=True), pd.concat([y_before, y_repair]).reset_index(drop=True)

            # Comparable Baselines

            new_all_retrained_model = get_sklearn_model(X_before_repair, y_before_repair, previous_model=mapped_model.model)
            new_all_retrained_accuracy = get_accuracy(new_all_retrained_model, X_test, y_test)
            new_all_retrained_accuracy_bump = new_all_retrained_accuracy - post_drift_test_accuracy

            new_drift_retrained_model = get_sklearn_model(X_repair, y_repair, previous_model=mapped_model.model)
            new_drift_retrained_accuracy = get_accuracy(new_drift_retrained_model, X_test, y_test)
            new_drift_retrained_accuracy_bump = new_drift_retrained_accuracy - post_drift_test_accuracy

            current_results_dict = {
                tester_constants.DATASET_NAME_COLUMN_NAME: file_name,
                tester_constants.DATASET_SIZE_COLUMN_NAME: len(dataset),
                tester_constants.TREE_SIZE_COLUMN_NAME: len(mapped_model),
                tester_constants.TREE_FEATURES_COUNT_COLUMN_NAME: len(mapped_model.model_used_features),
                tester_constants.REPAIR_WINDOW_PERCENTAGE_COLUMN_NAME: dataset.repair_proportion * dataset.repair_window_proportion * 100,
                tester_constants.REPAIR_WINDOW_SIZE_COLUMN_NAME: dataset.repair_window_size,
                tester_constants.DRIFT_SIZE_COLUMN_NAME: drift_size,
                tester_constants.TOTAL_DRIFT_TYPE_COLUMN_NAME: get_total_drift_types(drifted_features_types),
                tester_constants.DRIFT_SEVERITY_LEVEL_COLUMN_NAME: drift_severity_level,
                tester_constants.DRIFTED_FEATURES_COLUMN_NAME: ", ".join(map(lambda feature: f"{feature}: {faulty_features_nodes[feature]}", faulty_features_nodes)),
                tester_constants.DRIFTED_FEATURES_TYPES_COLUMN_NAME: get_drifted_features_types_string(drifted_features_types),
                tester_constants.DRIFT_DESCRIPTION_COLUMN_NAME: drift_description,
                tester_constants.ORIGINAL_ACCURACY_COLUMN_NAME: pre_drift_accuracy * 100,
                tester_constants.AFTER_ACCURACY_COLUMN_NAME: post_drift_test_accuracy * 100,
                tester_constants.AFTER_ACCURACY_DECREASE_COLUMN_NAME: post_drift_test_accuracy_drop * 100,
                f"{tester_constants.NEW_DRIFT_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_NAME_SUFFIX}": new_drift_retrained_accuracy * 100,
                f"{tester_constants.NEW_DRIFT_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_INCREASE_NAME_SUFFIX}": new_drift_retrained_accuracy_bump * 100,
                f"{tester_constants.NEW_ALL_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_NAME_SUFFIX}": new_all_retrained_accuracy * 100,
                f"{tester_constants.NEW_ALL_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_INCREASE_NAME_SUFFIX}": new_all_retrained_accuracy_bump * 100
            }

            for diagnoser_data in diagnosers_data:
                diagnoser_class_name = diagnoser_data["class_name"]
                diagnoser_output_name = diagnoser_data.get("output_name", diagnoser_class_name)
                diagnoser_parameters = diagnoser_data.get("parameters")
                diagnoser_class = get_diagnoser(diagnoser_class_name)
                if diagnoser_class is Oracle:
                    diagnoser_parameters = {"actual_faulty_features": drifted_features}
                else:
                    assert diagnoser_parameters is not None, "Diagnoser must have parameters property"
                diagnoser: ADiagnoser = diagnoser_class(mapped_model, X_repair, y_repair, **diagnoser_parameters)
                diagnoses: list[list[int]] = diagnoser.get_diagnoses()
                faulty_nodes_indices: list[int] = diagnoses[0]
                if diagnoser_class is not Oracle:
                    wasted_effort = get_wasted_effort(mapped_model, diagnoses, faulty_features_nodes)
                    faulty_nodes = [mapped_model[faulty_node_index] for faulty_node_index in faulty_nodes_indices]
                    detected_faulty_features = set([faulty_node.feature if not faulty_node.is_terminal() else "target" for faulty_node in faulty_nodes])
                    correctly_identified = get_correctly_identified_ratio(detected_faulty_features, drifted_features)
                    current_results_dict.update({
                        f"{diagnoser_output_name} {tester_constants.FAULTY_FEATURES_NAME_SUFFIX}": ", ".join(detected_faulty_features),
                        f"{diagnoser_output_name} {tester_constants.DIAGNOSES_NAME_SUFFIX}": ", ".join(map(str, diagnoses)),
                        f"{diagnoser_output_name} {tester_constants.WASTED_EFFORT_NAME_SUFFIX}": wasted_effort,
                        f"{diagnoser_output_name} {tester_constants.CORRECTLY_IDENTIFIED_NAME_SUFFIX}": correctly_identified * 100
                    })
                for fixer_data in fixers_data:
                    fixer_class_name = fixer_data["class_name"]
                    fixer_output_name = fixer_data.get("output_name", fixer_class_name)
                    fixer_parameters = fixer_data.get("parameters")
                    fixer_class = get_fixer(fixer_class_name)
                    fixer: AFixer = fixer_class(mapped_model, X_repair, y_repair, faulty_nodes_indices=faulty_nodes_indices, X_prior=X_repair, y_prior=y_repair, **fixer_parameters)
                    fixed_tree = fixer.fix_tree()
                    fixed_test_accuracy = get_accuracy(fixed_tree, X_test, y_test)
                    test_accuracy_bump = fixed_test_accuracy - post_drift_test_accuracy
                    current_results_dict.update({
                        f"{diagnoser_output_name}-{fixer_output_name} {tester_constants.FIX_ACCURACY_NAME_SUFFIX}": fixed_test_accuracy * 100,
                        f"{diagnoser_output_name}-{fixer_output_name} {tester_constants.FIX_ACCURACY_INCREASE_NAME_SUFFIX}": test_accuracy_bump * 100
                    })
            yield current_results_dict
        except Exception as e:
            exception_class = e.__class__.__name__
            diagnoser_info = f"(diagnoser: {diagnoser_class_name} - {diagnoser_output_name})" if 'diagnoser_output_name' in locals() else ""
            scenario_description = f"{drift_description}, repair window size: {dataset.repair_window_proportion} {diagnoser_info}"
            yield Exception(f"{exception_class} in {scenario_description}:\n "
                            f"{e}:\n "
                            f"{''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
                            f"Scenario: {scenario_description}"
                            )
            
def drift_tree_v2(dataset: Dataset,
                  repair_window_test_sizes: list[float] = tester_constants.REPAIR_WINDOW_TEST_SIZES,
                  min_drift_size: int = tester_constants.MIN_DRIFT_SIZE,
                  max_drift_size: int = tester_constants.MAX_DRIFT_SIZE
               ):
    """
    Generate a drifted in a multiple features
    """
    current_min_drift_size = max(min_drift_size, 1)
    current_max_drift_size = len(dataset.feature_types)
    if max_drift_size > 0:
        current_max_drift_size = min(current_max_drift_size, max_drift_size)
    for repair_window_test_size in repair_window_test_sizes:
        print(f"\tRepair window size: {repair_window_test_size}%")
        dataset.update_repair_window_size(repair_window_test_size)
        for drift_size in range(current_min_drift_size, current_max_drift_size + 1):
            print(f"\t\tDrift size: {drift_size} / {current_max_drift_size} features")
            for drifting_features in combinations(dataset.feature_types.keys(), drift_size):
                print(f"\t\t\tDrifting {', '.join(drifting_features)}")
                drifted_features_types = sorted([dataset.feature_types[drifting_feature] for drifting_feature in drifting_features])
                if not is_drift_contains_numeric_features(drifted_features_types): # TODO: Remove the condition
                    print(f"\t\t\t\tSkipping non-numeric drift: {', '.join(drifting_features)}")
                    continue
                for (X_before, y_before), (X_repair, y_repair), (X_test, y_test), drift_description in dataset.drift_generator_v2(drifting_features):
                    yield (X_before, y_before), (X_repair, y_repair), (X_test, y_test), drift_description, set(drifting_features), drifted_features_types, drift_size

def run_single_test_v2(directory, file_name, file_extension: str = ".csv", repair_window_test_sizes=tester_constants.REPAIR_WINDOW_TEST_SIZES, min_drift_size=tester_constants.MIN_DRIFT_SIZE, max_drift_size=tester_constants.MAX_DRIFT_SIZE, diagnosers_data: list[dict[str, object]] = tester_constants.DEFAULT_TESTING_DIAGNOSER, fixers_data: list[dict[str, object]] = tester_constants.DEFAULT_TESTING_FIXER):
    dataset = get_dataset(directory, file_name, file_extension=file_extension)
    
    for (X_before, y_before), (X_repair, y_repair), (X_test, y_test), drift_description, drifted_features, drifted_features_types, drift_size in drift_tree_v2(dataset, repair_window_test_sizes=repair_window_test_sizes, min_drift_size=min_drift_size, max_drift_size=max_drift_size):
        try:
            sklearn_model = get_sklearn_model(X_before, y_before)
            pre_drift_accuracy = sklearn_model.best_accuracy

            if pre_drift_accuracy < tester_constants.MINIMUM_ORIGINAL_ACCURACY:
                # Original model is not good enough
                continue
            
            mapped_model = get_mapped_model(sklearn_model, dataset.feature_types, X_before, y_before)

            if not all(map(lambda feature: feature in mapped_model.model_used_features, drifted_features)):    # redundant features drifted
                continue

            if X_repair.empty or X_test.empty:
                continue
            
            post_drift_test_accuracy = get_accuracy(mapped_model.model, X_test, y_test)
            post_drift_test_accuracy_drop = pre_drift_accuracy - post_drift_test_accuracy
            
            if pre_drift_accuracy - get_accuracy(mapped_model.model, X_repair, y_repair) < tester_constants.MINIMUM_DRIFT_ACCURACY_DROP or post_drift_test_accuracy_drop < tester_constants.MINIMUM_DRIFT_ACCURACY_DROP:    # insignificant drift
                continue

            faulty_features_nodes = get_drifted_nodes(mapped_model, drifted_features)

            print(f"\t\t\t\tDiagnosing")

            X_before_repair, y_before_repair = pd.concat([X_before, X_repair]).reset_index(drop=True), pd.concat([y_before, y_repair]).reset_index(drop=True)
            
            # Comparable Baselines

            new_all_retrained_model = get_sklearn_model(X_before_repair, y_before_repair, previous_model=mapped_model.model)
            new_all_retrained_accuracy = get_accuracy(new_all_retrained_model, X_test, y_test)
            new_all_retrained_accuracy_bump = new_all_retrained_accuracy - post_drift_test_accuracy

            new_drift_retrained_model = get_sklearn_model(X_repair, y_repair, previous_model=mapped_model.model)
            new_drift_retrained_accuracy = get_accuracy(new_drift_retrained_model, X_test, y_test)
            new_drift_retrained_accuracy_bump = new_drift_retrained_accuracy - post_drift_test_accuracy
            
            current_results_dict = {
                tester_constants.DATASET_NAME_COLUMN_NAME: file_name,
                tester_constants.DATASET_SIZE_COLUMN_NAME: len(dataset),
                tester_constants.TREE_SIZE_COLUMN_NAME: len(mapped_model),
                tester_constants.TREE_FEATURES_COUNT_COLUMN_NAME: len(mapped_model.model_used_features),
                tester_constants.REPAIR_WINDOW_PERCENTAGE_COLUMN_NAME: dataset.repair_proportion * dataset.repair_window_proportion * 100,
                tester_constants.REPAIR_WINDOW_SIZE_COLUMN_NAME: dataset.repair_window_size,
                tester_constants.DRIFT_SIZE_COLUMN_NAME: drift_size,
                tester_constants.TOTAL_DRIFT_TYPE_COLUMN_NAME: get_total_drift_types(drifted_features_types),
                tester_constants.DRIFT_SEVERITY_LEVEL_COLUMN_NAME: 1, # v2 does not support severity levels
                tester_constants.DRIFTED_FEATURES_COLUMN_NAME: ", ".join(map(lambda feature: f"{feature}: {faulty_features_nodes[feature]}", faulty_features_nodes)),
                tester_constants.DRIFTED_FEATURES_TYPES_COLUMN_NAME: get_drifted_features_types_string(drifted_features_types),
                tester_constants.DRIFT_DESCRIPTION_COLUMN_NAME: drift_description,
                tester_constants.ORIGINAL_ACCURACY_COLUMN_NAME: pre_drift_accuracy * 100,
                tester_constants.AFTER_ACCURACY_COLUMN_NAME: post_drift_test_accuracy * 100,
                tester_constants.AFTER_ACCURACY_DECREASE_COLUMN_NAME: post_drift_test_accuracy_drop * 100,
                f"{tester_constants.NEW_DRIFT_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_NAME_SUFFIX}": new_drift_retrained_accuracy * 100,
                f"{tester_constants.NEW_DRIFT_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_INCREASE_NAME_SUFFIX}": new_drift_retrained_accuracy_bump * 100,
                f"{tester_constants.NEW_ALL_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_NAME_SUFFIX}": new_all_retrained_accuracy * 100,
                f"{tester_constants.NEW_ALL_RETRAIN_COLUMNS_PREFIX} {tester_constants.FIX_ACCURACY_INCREASE_NAME_SUFFIX}": new_all_retrained_accuracy_bump * 100
            }

            for diagnoser_data in diagnosers_data:
                diagnoser_class_name = diagnoser_data["class_name"]
                diagnoser_output_name = diagnoser_data.get("output_name", diagnoser_class_name)
                diagnoser_parameters = diagnoser_data.get("parameters")
                diagnoser_class = get_diagnoser(diagnoser_class_name)
                if diagnoser_class is Oracle:
                    diagnoser_parameters = {"actual_faulty_features": drifted_features}
                else:
                    assert diagnoser_parameters is not None, "Diagnoser must have parameters property"
                diagnoser: ADiagnoser = diagnoser_class(mapped_model, X_repair, y_repair, **diagnoser_parameters)
                diagnoses: list[list[int]] = diagnoser.get_diagnoses()
                faulty_nodes_indices: list[int] = diagnoses[0]
                if diagnoser_class is not Oracle:
                    wasted_effort = get_wasted_effort(mapped_model, diagnoses, faulty_features_nodes)
                    faulty_nodes = [mapped_model[faulty_node_index] for faulty_node_index in faulty_nodes_indices]
                    detected_faulty_features = set([faulty_node.feature if not faulty_node.is_terminal() else "target" for faulty_node in faulty_nodes])
                    correctly_identified = get_correctly_identified_ratio(detected_faulty_features, drifted_features)
                    current_results_dict.update({
                        f"{diagnoser_output_name} {tester_constants.FAULTY_FEATURES_NAME_SUFFIX}": ", ".join(detected_faulty_features),
                        f"{diagnoser_output_name} {tester_constants.DIAGNOSES_NAME_SUFFIX}": ", ".join(map(str, diagnoses)),
                        f"{diagnoser_output_name} {tester_constants.WASTED_EFFORT_NAME_SUFFIX}": wasted_effort,
                        f"{diagnoser_output_name} {tester_constants.CORRECTLY_IDENTIFIED_NAME_SUFFIX}": correctly_identified * 100
                    })
                for fixer_data in fixers_data:
                    fixer_class_name = fixer_data["class_name"]
                    fixer_output_name = fixer_data.get("output_name", fixer_class_name)
                    fixer_parameters = fixer_data.get("parameters")
                    fixer_class = get_fixer(fixer_class_name)
                    fixer: AFixer = fixer_class(mapped_model, X_repair, y_repair, faulty_nodes_indices=faulty_nodes_indices, X_prior=X_repair, y_prior=y_repair, **fixer_parameters)
                    fixed_tree = fixer.fix_tree()
                    fixed_test_accuracy = get_accuracy(fixed_tree, X_test, y_test)
                    test_accuracy_bump = fixed_test_accuracy - post_drift_test_accuracy
                    current_results_dict.update({
                        f"{diagnoser_output_name}-{fixer_output_name} {tester_constants.FIX_ACCURACY_NAME_SUFFIX}": fixed_test_accuracy * 100,
                        f"{diagnoser_output_name}-{fixer_output_name} {tester_constants.FIX_ACCURACY_INCREASE_NAME_SUFFIX}": test_accuracy_bump * 100
                    })
            yield current_results_dict
        except Exception as e:
            exception_class = e.__class__.__name__
            diagnoser_info = f"(diagnoser: {diagnoser_class_name} - {diagnoser_output_name})" if 'diagnoser_output_name' in locals() else ""
            scenario_description = f"{drift_description}, repair window size: {dataset.repair_window_proportion} {diagnoser_info}"
            yield Exception(f"{exception_class} in {scenario_description}:\n "
                            f"{e}:\n "
                            f"{''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
                            f"Scenario: {scenario_description}"
                            )

def get_example_mapped_model(directory=tester_constants.DATASETS_DIRECTORY_FULL_PATH, file_name=tester_constants.EXAMPLE_FILE_NAME):
    dataset = get_dataset(directory, file_name)
    X_train, y_train = dataset.get_before_concept_data()
    sklearn_model = get_sklearn_model(X_train, y_train)
    return get_mapped_model(sklearn_model, dataset.feature_types, X_train, y_train)

def sanity_run(directory=tester_constants.DATASETS_DIRECTORY_FULL_PATH, file_name=tester_constants.EXAMPLE_FILE_NAME, diagnosers_data=tester_constants.DEFAULT_TESTING_DIAGNOSER):
    for result in run_single_test(directory=directory, file_name=file_name, diagnosers_data=diagnosers_data):
        print(result)
        
if __name__ == "__main__":
    file_name = tester_constants.EXAMPLE_FILE_NAME if len(sys.argv) < 2 else sys.argv[1]
    sanity_run(tester_constants.DATASETS_DIRECTORY_FULL_PATH, file_name)