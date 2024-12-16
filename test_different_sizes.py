from os.path import exists as os_path_exists
from os import mkdir as os_mkdir
from pandas import DataFrame
from csv import DictReader

from APPETITE.Constants import TEST_PROPORTION
from Tester import *

DATA_DIRECTORY = "data"
DATASET_DESCRIPTION_FILE = "all_datasets.csv"
DATASETS_DIRECTORY = "Classification_Datasets"
DATASETS_FULL_PATH = f"{DATA_DIRECTORY}\\{DATASETS_DIRECTORY}\\"
RESULTS_DIRECTORY = "results"
RESULTS_FULL_PATH = f"{DATA_DIRECTORY}\\{RESULTS_DIRECTORY}\\"

SIZES_COLUMN_NAME = "train-after-test"
COUNT_COLUMN_NAME = "drifts count"
WASTED_EFFORT_NAME_SUFFIX = " wasted effort"
FIX_ACCURACY_NAME_SUFFIX = " fix accuracy percentage"
FIX_ACCURACY_INCREASE_NAME_SUFFIX = " fix accuracy increase percentage"

total_results_columns = ["after accuracy decrease percentage", "after retrain accuracy", "before after retrain accuracy"]

if isinstance(DEFAULT_TESTING_DIAGNOSER, str):
    DEFAULT_TESTING_DIAGNOSER = (DEFAULT_TESTING_DIAGNOSER, )

for diagnoser_name in DEFAULT_TESTING_DIAGNOSER:
    total_results_columns.append(diagnoser_name + WASTED_EFFORT_NAME_SUFFIX)
    total_results_columns.append(diagnoser_name + FIX_ACCURACY_NAME_SUFFIX)
    total_results_columns.append(diagnoser_name + FIX_ACCURACY_INCREASE_NAME_SUFFIX)

# Create DataFrame for the aggregated results
total_results = DataFrame(columns=[SIZES_COLUMN_NAME, COUNT_COLUMN_NAME] + total_results_columns)

TRAIN_AFTER_PERCENTAGE_SIZE = int((1 - TEST_PROPORTION) * 100)

for train_size in range(5, TRAIN_AFTER_PERCENTAGE_SIZE, 5):
    with open(f"{DATA_DIRECTORY}/{DATASET_DESCRIPTION_FILE}", "r") as descriptions_file:
        train_proportion, after_proportion = train_size / 100, (TRAIN_AFTER_PERCENTAGE_SIZE - train_size) / 100

        current_total_row_dict = {SIZES_COLUMN_NAME: f"{train_proportion}-{after_proportion}-{TEST_PROPORTION}"
                                  , COUNT_COLUMN_NAME: 0}
        current_total_row_dict.update({total_results_column: 0 for total_results_column in total_results_columns})
        
        current_total_row_dict[COUNT_COLUMN_NAME] = 0
        
        proportion_tuple = (train_proportion, after_proportion, TEST_PROPORTION)

        descriptions_reader = DictReader(descriptions_file)
        for dataset_description in descriptions_reader:
            dataset_name = dataset_description["name"]
            
            for test_result in run_test(DATASETS_FULL_PATH, dataset_name + ".csv", proportions_tuple=proportion_tuple):
                current_total_row_dict[COUNT_COLUMN_NAME] = current_total_row_dict[COUNT_COLUMN_NAME] + 1
                for key, value in test_result.items():
                    if key in total_results_columns:
                        current_total_row_dict[key] += value
        
        current_total_row_dict.update({total_results_column: current_total_row_dict[total_results_column] / current_total_row_dict[COUNT_COLUMN_NAME] for total_results_column in total_results_columns})
        total_results = total_results._append(current_total_row_dict, ignore_index=True)

if not os_path_exists(RESULTS_FULL_PATH):
    os_mkdir(RESULTS_FULL_PATH)
    
total_results.to_csv(f"{RESULTS_FULL_PATH}total_results.csv", index=False)