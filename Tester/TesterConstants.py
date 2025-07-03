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

DATA_DIRECTORY = "data"
DATASET_DESCRIPTION_FILE_NAME = "all_datasets"
DATASET_DESCRIPTION_FILE_PATH = os_path.join(DATA_DIRECTORY, f"{DATASET_DESCRIPTION_FILE_NAME}.csv")
DATASETS_DIRECTORY = "Classification_Datasets"
DATASETS_FULL_PATH = os_path.join(DATA_DIRECTORY, DATASETS_DIRECTORY)

RESULTS_DIRECTORY = "results"
RESULTS_FULL_PATH = os_path.join(DATA_DIRECTORY, RESULTS_DIRECTORY)
TEMP_RESULTS_DIRECTORY = "temp"
TEMP_RESULTS_FULL_PATH = os_path.join(RESULTS_FULL_PATH, TEMP_RESULTS_DIRECTORY)
RESULTS_FILE_NAME_PREFIX = "results"
ERRORS_FILE_NAME_PREFIX = "errors"

EXAMPLE_FILE_NAME = "bank"

MIN_DRIFT_SIZE = 1 # min amount of features to drift
MAX_DRIFT_SIZE = 4 # max amount of features to drift, -1 means all features

AFTER_WINDOW_TEST_SIZES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]