from os.path import exists as os_path_exists
from os import mkdir as os_mkdir
from pandas import DataFrame
from csv import DictReader
from time import time

from Tester import *

start_time = time()

DATA_DIRECTORY = "data"
DATASET_DESCRIPTION_FILE = "all_datasets.csv"
DATASETS_DIRECTORY = "Classification_Datasets"
DATASETS_FULL_PATH = f"{DATA_DIRECTORY}\\{DATASETS_DIRECTORY}\\"
RESULTS_DIRECTORY = "results"
RESULTS_FULL_PATH = f"{DATA_DIRECTORY}\\{RESULTS_DIRECTORY}\\"

AFTER_WINDOW_COLUMN_NAME = "after window size"
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

if __name__ == "__main__":
    # Create DataFrame for the aggregated results
    total_results = DataFrame(columns=[AFTER_WINDOW_COLUMN_NAME, COUNT_COLUMN_NAME] + total_results_columns)

    for after_window_percentage in range(5, 101, 5):
        with open(f"{DATA_DIRECTORY}/{DATASET_DESCRIPTION_FILE}", "r") as descriptions_file:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", f"CURRENT AFTER WINDOW SIZE: {after_window_percentage}%", "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

            current_total_row_dict = {AFTER_WINDOW_COLUMN_NAME: f"{after_window_percentage}%"
                                    , COUNT_COLUMN_NAME: 0}
            current_total_row_dict.update({total_results_column: 0 for total_results_column in total_results_columns})

            descriptions_reader = DictReader(descriptions_file)
            for dataset_description in descriptions_reader:
                dataset_name = dataset_description["name"]
                
                try:
                    for test_result in run_test(DATASETS_FULL_PATH, dataset_name + ".csv", after_window_size=after_window_percentage / 100):
                        current_total_row_dict[COUNT_COLUMN_NAME] = current_total_row_dict[COUNT_COLUMN_NAME] + 1
                        for key, value in test_result.items():
                            if key in total_results_columns:
                                current_total_row_dict[key] += value
                except Exception as e:
                    if isinstance(e, KeyError):
                        raise(e)
                    continue
            
            current_total_row_dict.update({total_results_column: current_total_row_dict[total_results_column] / current_total_row_dict[COUNT_COLUMN_NAME] for total_results_column in total_results_columns})
            total_results = total_results._append(current_total_row_dict, ignore_index=True)

    if not os_path_exists(RESULTS_FULL_PATH):
        os_mkdir(RESULTS_FULL_PATH)
        
    total_results.to_csv(f"{RESULTS_FULL_PATH}total_results.csv", index=False)

    print(f"Total time: {time() - start_time}")