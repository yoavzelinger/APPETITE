import os
from csv import DictReader
from pandas import DataFrame

from Tester import *

DATA_DIRECTORY = "data"
DATASET_DESCRIPTION_FILE = "all_datasets.csv"
DATASETS_DIRECTORY = "Classification_Datasets"
DATASETS_FULL_PATH = f"{DATA_DIRECTORY}\\{DATASETS_DIRECTORY}\\"
RESULTS_DIRECTORY = "results"
RESULTS_FULL_PATH = f"{DATA_DIRECTORY}\\{RESULTS_DIRECTORY}\\"

FAULTY_NODE_NAME_SUFFIX = " faulty node index"
FAULTY_FEATURE_NAME_SUFFIX = " faulty feature"
FIX_ACCURACY_NAME_SUFFIX = " fix accuracy increase percentage"
AVERAGE_FIX_ACCURACY_NAME = " average" + FIX_ACCURACY_NAME_SUFFIX

if isinstance(DEFAULT_TESTING_DIAGNOSER, str):
    DEFAULT_TESTING_DIAGNOSER = (DEFAULT_TESTING_DIAGNOSER, )

raw_results_columns = ["drift description", "tree size", "after accuracy decrease percentage"]
aggregated_groupby_columns = ["name", "tree size", "drifts count", "average after accuracy decrease percentage"]

aggregated_summarizes_columns = []
for diagnoser_name in DEFAULT_TESTING_DIAGNOSER:
    raw_results_columns.append(diagnoser_name + FAULTY_NODE_NAME_SUFFIX)
    raw_results_columns.append(diagnoser_name + FAULTY_FEATURE_NAME_SUFFIX)
    raw_results_columns.append(diagnoser_name + FIX_ACCURACY_NAME_SUFFIX)
    aggregated_summarizes_columns.append(diagnoser_name + AVERAGE_FIX_ACCURACY_NAME)

# Create DataFrame for the aggregated results
raw_results = DataFrame(columns=raw_results_columns)
aggregated_results = DataFrame(columns=aggregated_groupby_columns + aggregated_summarizes_columns)
errors = DataFrame(columns=["name", "error"])

with open(f"{DATA_DIRECTORY}/{DATASET_DESCRIPTION_FILE}", "r") as descriptions_file:
    descriptions_reader = DictReader(descriptions_file)
    for dataset_description in descriptions_reader:
        dataset_name = dataset_description["name"]
        drifts_count = 0

        current_aggregated_row_dict = {
            "name": dataset_name,
            "tree size": -1,
            "average after accuracy decrease percentage": 0
        }
        current_aggregated_row_dict.update({summarize_column_name: 0 for summarize_column_name in aggregated_summarizes_columns})

        print(f"Running tests for {dataset_name}")
        try:
            for test_result in run_test(DATASETS_FULL_PATH, dataset_name + ".csv"):
                drifts_count += 1
                current_aggregated_row_dict["tree size"] = test_result["tree size"]
                current_aggregated_row_dict["average after accuracy decrease percentage"] = test_result["after accuracy decrease percentage"]
                for diagnoser_name in DEFAULT_TESTING_DIAGNOSER:
                    current_aggregated_row_dict[diagnoser_name + AVERAGE_FIX_ACCURACY_NAME] += test_result[diagnoser_name + FIX_ACCURACY_NAME_SUFFIX]
                raw_results = raw_results._append(test_result, ignore_index=True)
            if drifts_count == 0:
                continue
            current_aggregated_row_dict["drifts count"] = drifts_count
            for summarize_column_name in aggregated_summarizes_columns:
                current_aggregated_row_dict[summarize_column_name] /= drifts_count
            aggregated_results = aggregated_results._append(current_aggregated_row_dict, ignore_index=True)
        except Exception as e:
            errors = errors._append({"name": dataset_name, "error": str(e)}, ignore_index=True)
            continue

aggregating_total_row = {
    "name": "TOTAL",
    "tree size": aggregated_results["tree size"].mean(),
    "drifts count": aggregated_results["drifts count"].mean(),
    "average after accuracy decrease percentage": aggregated_results["average after accuracy decrease percentage"].mean()
}
for diagnoser_name in DEFAULT_TESTING_DIAGNOSER:
    aggregating_total_row[diagnoser_name + AVERAGE_FIX_ACCURACY_NAME] = raw_results[diagnoser_name + FIX_ACCURACY_NAME_SUFFIX].mean()

aggregated_results = aggregated_results._append(aggregating_total_row, ignore_index=True)

if not os.path.exists(RESULTS_FULL_PATH):
    os.mkdir(RESULTS_FULL_PATH)

aggregated_results.to_csv(f"{RESULTS_FULL_PATH}/aggregated_results.csv", index=False)
raw_results.to_csv(f"{RESULTS_FULL_PATH}/all_results.csv", index=False)
if not errors.empty:
    errors.to_csv(f"{RESULTS_FULL_PATH}/errors.csv", index=False)

print("All tests are done! average accuracy incremental:")
for diagnoser_name in DEFAULT_TESTING_DIAGNOSER:
    print(f"{diagnoser_name}: {aggregating_total_row[diagnoser_name + AVERAGE_FIX_ACCURACY_NAME]}%")