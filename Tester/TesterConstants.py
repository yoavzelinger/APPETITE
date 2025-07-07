from os import path as os_path
import APPETITE.Constants as constants

MINIMUM_ORIGINAL_ACCURACY = 0.75
MINIMUM_DRIFT_ACCURACY_DROP = 0.1

DEFAULT_TESTING_DIAGNOSER = {
    "output_name": "Regular_SFLDT",
	"class_name": "SFLDT",
	"parameters": {}
    }

if isinstance(DEFAULT_TESTING_DIAGNOSER, dict):
    DEFAULT_TESTING_DIAGNOSER = (DEFAULT_TESTING_DIAGNOSER, )
assert isinstance(DEFAULT_TESTING_DIAGNOSER, (list, tuple)) and all(isinstance(diagnoser_data, dict) for diagnoser_data in DEFAULT_TESTING_DIAGNOSER), \
    "DEFAULT_FIXING_DIAGNOSER must be a tuple of dictionaries, each dictionary representing a diagnoser."

TESTING_DIAGNOSERS_CONFIGURATION_FILE_NAME = "TestingDiagnosersData"

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

OUTPUT_DIRECTORY_NAME = "results"
OUTPUT_DIRECTORY_FULL_PATH = os_path.join(DATA_DIRECTORY_NAME, OUTPUT_DIRECTORY_NAME)
TEMP_OUTPUT_DIRECTORY_NAME = "temp"
TEMP_OUTPUT_DIRECTORY_FULL_PATH = os_path.join(OUTPUT_DIRECTORY_FULL_PATH, TEMP_OUTPUT_DIRECTORY_NAME)
RESULTS_FILE_NAME_PREFIX = "results"
EMPTY_RESULTS_FILE_NAME_PREFIX = "EMPTY"
ERRORS_FILE_NAME_PREFIX = "ERRORS"


EXAMPLE_FILE_NAME = "bank"

MIN_DRIFT_SIZE = 1 # min amount of features to drift
MAX_DRIFT_SIZE = 4 # max amount of features to drift, -1 means all features

AFTER_WINDOW_TEST_SIZES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]