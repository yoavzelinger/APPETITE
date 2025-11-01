from os import path as os_path
from json import load as load_json
from datetime import datetime

import APPETITE.Constants as constants

from APPETITE.Diagnosers import Oracle
from APPETITE.Fixers import *

RANDOM_STATE = constants.RANDOM_STATE

# Dataset partitions sizes
BEFORE_PROPORTION = 0.7
REPAIR_PROPORTION = 0.1
TEST_PROPORTION = 0.2
PROPORTIONS_TUPLE = (BEFORE_PROPORTION, REPAIR_PROPORTION, TEST_PROPORTION)
DEFAULT_REPAIR_WINDOW_PROPORTION = 1

DRIFT_SYNTHESIZING_VERSION = 1

VALIDATION_SIZE = 0.2

# one hot encode categorical features to modify decision trees creation
ONE_HOT_ENCODING_CATEGORICAL = False

# Drift severity levels
NUMERIC_DRIFT_SEVERITIES = {
    1: (-0.5, 0.5),
    2: (-1, 1),
    3: (-2, 2)
}
CATEGORICAL_DRIFT_SEVERITIES = {
    1: (0.3, ),
    2: (0.5, 0.7),
    3: (0.9, )
}
DEFAULT_GENERATED_SEVERITY_LEVELS = (1, 2, 3)

# Grid search parameters
CROSS_VALIDATION_SPLIT_COUNT = 5
_CRITERIONS = ["gini", "entropy"]
_MAX_LEAF_NODES = [10, 20, 30]
PARAM_GRID = {
    "criterion": _CRITERIONS,
    "max_leaf_nodes": _MAX_LEAF_NODES
}

MINIMUM_ORIGINAL_ACCURACY = 0.75
MINIMUM_DRIFT_ACCURACY_DROP = 0.1

SKIP_EXCEPTIONS = False

WASTED_EFFORT_REQUIRE_FULL_FIX = True # Fix all faulty features

# How to handle "healthy" components did not appear in any diagnoses.
WASTED_EFFORT_MISSING_ACTIONS = [
    "all", # Add all as wasted effort
    "none", # Do not add any as wasted effort
    "random", # Randomly choose components until all faults were fixed.
              # Since Since we already fixed current_wasted_effort components and there are undetected_faults_count left to fix, 
              # we can calculate the wasted effort as:
              #                             undetected_faults_count * (healthy_nodes_counts - current_wasted_effort)
              #                         --------------------------------------------------------------------------------
              #                                                     undetected_faults_count + 1
]
WASTED_EFFORT_MISSING_ACTION = "random"
assert WASTED_EFFORT_MISSING_ACTION in WASTED_EFFORT_MISSING_ACTIONS, f"WASTED_EFFORT_MISSING_ACTION must be one of {WASTED_EFFORT_MISSING_ACTIONS}, got {WASTED_EFFORT_MISSING_ACTION}."

DATA_DIRECTORY_NAME = "data"

DATASET_DESCRIPTION_FILE_NAME = "all_datasets"
DATASET_DESCRIPTION_FILE_PATH = os_path.join(DATA_DIRECTORY_NAME, f"{DATASET_DESCRIPTION_FILE_NAME}.csv")

DATASETS_DIRECTORY_NAME = "Classification_Datasets"
DATASETS_DIRECTORY_FULL_PATH = os_path.join(DATA_DIRECTORY_NAME, DATASETS_DIRECTORY_NAME)

OUTPUT_DIRECTORY_FULL_PATH = "results"
TEMP_OUTPUT_DIRECTORY_NAME = "temp"
TEMP_OUTPUT_DIRECTORY_FULL_PATH = os_path.join(OUTPUT_DIRECTORY_FULL_PATH, TEMP_OUTPUT_DIRECTORY_NAME)
RESULTS_FILE_NAME_PREFIX = "results"
EMPTY_RESULTS_FILE_NAME_PREFIX = "EMPTY"
ERRORS_FILE_NAME_PREFIX = "ERRORS"
DEFAULT_RESULTS_FILENAME_PREFIX = "time"
DEFAULT_RESULTS_FILENAME_EXTENDED_PREFIX = f"{DEFAULT_RESULTS_FILENAME_PREFIX}_{datetime.now().strftime('%d-%m_%H-%M-%S')}" # Unique file prefix

MERGED_RESULTS_SHEET_NAME = "merged"
RAW_RESULTS_SHEET_NAME = "raw"

EXAMPLE_FILE_NAME = "bank"

MIN_DRIFT_SIZE = 1 # min amount of features to drift
MAX_DRIFT_SIZE = 4 # max amount of features to drift, -1 means all features

REPAIR_WINDOW_TEST_SIZES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Testing Columns

#   DEFAULT TESTING DIAGNOSER
DEFAULT_TESTING_DIAGNOSER = {
        "output_name": "SFLDT (BASELINE)",
        "class_name": "SFLDT",
        "parameters": {
            "use_tests_confidence": False
        }
    }

if isinstance(DEFAULT_TESTING_DIAGNOSER, dict):
    DEFAULT_TESTING_DIAGNOSER = [DEFAULT_TESTING_DIAGNOSER]
assert isinstance(DEFAULT_TESTING_DIAGNOSER, list) and all(isinstance(diagnoser_data, dict) for diagnoser_data in DEFAULT_TESTING_DIAGNOSER), \
    "DEFAULT_FIXING_DIAGNOSER must be a tuple of dictionaries, each dictionary representing a diagnoser."

#   TESTING DIAGNOSERS DATA
TESTING_DIAGNOSERS_CONFIGURATION_FILE_NAME = "TestingDiagnosersData"
diagnosers_data = {}
with open(os_path.join(__package__, f"{TESTING_DIAGNOSERS_CONFIGURATION_FILE_NAME}.json"), "r") as testing_diagnosers_configuration_file:
    diagnosers_data = load_json(testing_diagnosers_configuration_file)
diagnosers_data: list[dict[str, object]] = [diagnoser_data for diagnoser_data in diagnosers_data if not diagnoser_data.get("disabled", False)]
diagnosers_output_names = list(map(lambda diagnoser_data: diagnoser_data.get("output_name", diagnoser_data["class_name"]), diagnosers_data))
TESTED_FIXERS: list[AFixer] = [AllNodesFixer, TopNodeFixer, TopFeaturesNodesFixer, SubTreeRetrainingFixer]
fixers_output_names = list(map(lambda fixer: fixer.alias, TESTED_FIXERS))


#   TESTING INFO COLUMNS
DATASET_NAME_COLUMN_NAME = "dataset name"
DATASET_SIZE_COLUMN_NAME = "dataset size"
TREE_SIZE_COLUMN_NAME = "tree size"
TREE_FEATURES_COUNT_COLUMN_NAME = "tree features count"
REPAIR_WINDOW_PERCENTAGE_COLUMN_NAME = "repair window (%)"
REPAIR_WINDOW_SIZE_COLUMN_NAME = "repair window size"
DRIFT_SIZE_COLUMN_NAME = "drift size"
TOTAL_DRIFT_TYPE_COLUMN_NAME = "total drift type"
DRIFT_SEVERITY_LEVEL_COLUMN_NAME = "drift severity level"
DRIFTED_FEATURES_COLUMN_NAME = "drifted features"
DRIFTED_FEATURES_TYPES_COLUMN_NAME = "drifted features types"
DRIFT_DESCRIPTION_COLUMN_NAME = "drift description"

GROUP_BY_COLUMNS = {
    DATASET_NAME_COLUMN_NAME: "string",
    DATASET_SIZE_COLUMN_NAME: "int64",
    TREE_SIZE_COLUMN_NAME: "int64",
    TREE_FEATURES_COUNT_COLUMN_NAME: "int64",
    REPAIR_WINDOW_PERCENTAGE_COLUMN_NAME: "float64",
    REPAIR_WINDOW_SIZE_COLUMN_NAME: "int64",
    DRIFT_SIZE_COLUMN_NAME: "int64",
    TOTAL_DRIFT_TYPE_COLUMN_NAME: "string",
    DRIFT_SEVERITY_LEVEL_COLUMN_NAME: "int64",
}
GROUP_BY_COLUMN_NAMES = list(GROUP_BY_COLUMNS.keys())

DRIFT_DESCRIBING_COLUMNS = {
    DRIFTED_FEATURES_COLUMN_NAME: "string",
    DRIFTED_FEATURES_TYPES_COLUMN_NAME: "string",
    DRIFT_DESCRIPTION_COLUMN_NAME: "string",
}

#   COMMON RESULTS COLUMNS
ORIGINAL_ACCURACY_COLUMN_NAME = "original accuracy"
AFTER_ACCURACY_COLUMN_NAME = "after accuracy"
AFTER_ACCURACY_DECREASE_COLUMN_NAME = "after accuracy decrease"

#   METRICS SUFFIXES
FAULTY_FEATURES_NAME_SUFFIX = "faulty features"
DIAGNOSES_NAME_SUFFIX = "diagnoses"
WASTED_EFFORT_NAME_SUFFIX = "wasted-effort"
CORRECTLY_IDENTIFIED_NAME_SUFFIX = "correctly-identified"
FIX_ACCURACY_NAME_SUFFIX = "fix accuracy"
FIX_ACCURACY_INCREASE_NAME_SUFFIX = "fix accuracy increase"

#   RETRAINING PREFIXES
NEW_DRIFT_RETRAIN_COLUMNS_PREFIX = "new-drift-retrain"
NEW_ALL_RETRAIN_COLUMNS_PREFIX = "new-all-retrain"

BASELINES_OUTPUT_NAMES = [NEW_DRIFT_RETRAIN_COLUMNS_PREFIX, NEW_ALL_RETRAIN_COLUMNS_PREFIX]

METRICS_COLUMNS = {
    ORIGINAL_ACCURACY_COLUMN_NAME: "float64",
    AFTER_ACCURACY_COLUMN_NAME: "float64",
    AFTER_ACCURACY_DECREASE_COLUMN_NAME: "float64"
}
for baseline_output_name in BASELINES_OUTPUT_NAMES:
    METRICS_COLUMNS[f"{baseline_output_name} {FIX_ACCURACY_NAME_SUFFIX}"] = "float64"
    METRICS_COLUMNS[f"{baseline_output_name} {FIX_ACCURACY_INCREASE_NAME_SUFFIX}"] = "float64"

for diagnoser_output_name in diagnosers_output_names:
    if diagnoser_output_name != Oracle.__name__: 
        METRICS_COLUMNS[f"{diagnoser_output_name} {DIAGNOSES_NAME_SUFFIX}"] = "string"
        METRICS_COLUMNS[f"{diagnoser_output_name} {WASTED_EFFORT_NAME_SUFFIX}"] = "float64"
        METRICS_COLUMNS[f"{diagnoser_output_name} {FAULTY_FEATURES_NAME_SUFFIX}"] = "string"
        METRICS_COLUMNS[f"{diagnoser_output_name} {CORRECTLY_IDENTIFIED_NAME_SUFFIX}"] = "float64"
    for fixer_output_name in fixers_output_names:
        METRICS_COLUMNS[f"{diagnoser_output_name}-{fixer_output_name} {FIX_ACCURACY_NAME_SUFFIX}"] = "float64"
        METRICS_COLUMNS[f"{diagnoser_output_name}-{fixer_output_name} {FIX_ACCURACY_INCREASE_NAME_SUFFIX}"] = "float64"


RAW_RESULTS_COLUMNS = GROUP_BY_COLUMNS | DRIFT_DESCRIBING_COLUMNS | METRICS_COLUMNS
RAW_RESULTS_COLUMN_NAMES = list(RAW_RESULTS_COLUMNS.keys())


#   MERGE RESULTS INFO
AGGREGATED_TESTS_COUNT_COLUMN = DRIFT_DESCRIPTION_COLUMN_NAME
TESTS_COUNTS_COLUMN_NAME = "tests count"

AGGREGATED_METRICS_COLUMNS = {metric_column_name: metric_column_dtype for metric_column_name, metric_column_dtype in METRICS_COLUMNS.items() if metric_column_dtype != "string"}
EXTENDED_METRICS_COLUMNS = {TESTS_COUNTS_COLUMN_NAME: "int64"} | AGGREGATED_METRICS_COLUMNS
EXTENDED_METRICS_COLUMN_NAMES = list(EXTENDED_METRICS_COLUMNS.keys())