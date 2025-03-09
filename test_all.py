import os
from sys import exit
from argparse import ArgumentParser
from datetime import datetime
from csv import DictReader
from pandas import DataFrame

from Tester import *

from warnings import simplefilter as warnings_simplefilter
warnings_simplefilter(action='ignore', category=FutureWarning)

FAULTY_NODES_NAME_SUFFIX = " faulty nodes indicies"
FAULTY_FEATURES_NAME_SUFFIX = " faulty features"
WASTED_EFFORT_NAME_SUFFIX = " wasted effort"
AVERAGE_WASTED_EFFORT_NAME_SUFFIX = " average" + WASTED_EFFORT_NAME_SUFFIX
FIX_ACCURACY_NAME_SUFFIX = " fix accuracy"
AVERAGE_FIX_ACCURACY_NAME_SUFFIX = " average" + FIX_ACCURACY_NAME_SUFFIX
FIX_ACCURACY_INCREASE_NAME_SUFFIX = " fix accuracy increase"
AVERAGE_FIX_ACCURACY_INCREASE_NAME_SUFFIX = " average" + FIX_ACCURACY_INCREASE_NAME_SUFFIX

parser = ArgumentParser(description="Run all tests")
parser.add_argument("-o", "--output", type=str, help="Output file name prefix, default is the result_TIMESTAMP", default=f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")
parser.add_argument("-d", "--diagnosers", type=str, nargs="+", help=f"List of diagnosers, default is {DEFAULT_TESTING_DIAGNOSER}", default=DEFAULT_TESTING_DIAGNOSER)
parser.add_argument("-s", "--stop", action="store_true", help="Stop on exception, default is false and writing the errors to errors file", default=STOP_ON_EXCEPTION)
parser.add_argument("-c", "--count", type=int, help="Number of tests to run, default is running all", default=-1)
parser.add_argument("-n", "--name", type=str, help="Name of the dataset to run, use only if you want to run a specific test")

args = parser.parse_args()
RESULTS_FILE_PATH_PREFIX, ERRORS_FILE_PATH_PREFIX = f"{RESULTS_FILE_PATH_PREFIX}_{args.output}", f"{ERRORS_FILE_PATH_PREFIX}_{args.output}"
DEFAULT_TESTING_DIAGNOSER = args.diagnosers
print(f"Running tests with diagnosers: {DEFAULT_TESTING_DIAGNOSER}")
STOP_ON_EXCEPTION = args.stop
datasets_count = args.count
if args.name:
    single_test.sanity_run(file_name=args.name + ".csv")
    exit(0)

raw_results_columns = ["drift description", "tree size", "after accuracy decrease", "after retrain accuracy", "after retrain accuracy increase", "before after retrain accuracy", "before after retrain accuracy increase"]
aggregated_groupby_columns = ["name", "tree size", "drifts count"]

aggregated_summarizes_columns = ["average after accuracy decrease", "average after retrain accuracy", "average after retrain accuracy increase", "average before after retrain accuracy", "average before after retrain accuracy increase"]
for diagnoser_name in DEFAULT_TESTING_DIAGNOSER:
    raw_results_columns.append(diagnoser_name + FAULTY_NODES_NAME_SUFFIX)
    raw_results_columns.append(diagnoser_name + FAULTY_FEATURES_NAME_SUFFIX)
    raw_results_columns.append(diagnoser_name + WASTED_EFFORT_NAME_SUFFIX)
    raw_results_columns.append(diagnoser_name + FIX_ACCURACY_NAME_SUFFIX)
    raw_results_columns.append(diagnoser_name + FIX_ACCURACY_INCREASE_NAME_SUFFIX)
    aggregated_summarizes_columns.append(diagnoser_name + AVERAGE_WASTED_EFFORT_NAME_SUFFIX)
    aggregated_summarizes_columns.append(diagnoser_name + AVERAGE_FIX_ACCURACY_NAME_SUFFIX)
    aggregated_summarizes_columns.append(diagnoser_name + AVERAGE_FIX_ACCURACY_INCREASE_NAME_SUFFIX)

# Create DataFrame for the aggregated results
raw_results = DataFrame(columns=raw_results_columns)
aggregated_results = DataFrame(columns=aggregated_groupby_columns + aggregated_summarizes_columns)
errors = DataFrame(columns=["name", "error"])

with open(DATASET_DESCRIPTION_FILE_PATH, "r") as descriptions_file:
    descriptions_reader = DictReader(descriptions_file)
    for dataset_description in descriptions_reader:
        if not datasets_count:
            break
        datasets_count -= 1
        dataset_name = dataset_description["name"]
        drifts_count = 0

        current_aggregated_row_dict = {
            "name": dataset_name,
            "tree size": -1,
            "average after accuracy decrease": 0,
            "average after retrain accuracy": 0,
            "average after retrain accuracy increase": 0,
            "average before after retrain accuracy": 0,
            "average before after retrain accuracy increase": 0
        }
        current_aggregated_row_dict.update({summarize_column_name: 0 for summarize_column_name in aggregated_summarizes_columns})
        if dataset_name in ("image-segmentation", "car"):
            continue
        print(f"Running tests for {dataset_name}")
        try:
            for test_result in run_single_test(DATASETS_FULL_PATH, dataset_name + ".csv", diagnoser_names=DEFAULT_TESTING_DIAGNOSER):
                drifts_count += 1
                current_aggregated_row_dict["tree size"] = test_result["tree size"]
                current_aggregated_row_dict["average after accuracy decrease"] += test_result["after accuracy decrease"]
                current_aggregated_row_dict["average after retrain accuracy"] += test_result["after retrain accuracy"]
                current_aggregated_row_dict["average after retrain accuracy increase"] += test_result["after retrain accuracy increase"]
                current_aggregated_row_dict["average before after retrain accuracy"] += test_result["before after retrain accuracy"]
                current_aggregated_row_dict["average before after retrain accuracy increase"] += test_result["before after retrain accuracy increase"]
                for diagnoser_name in DEFAULT_TESTING_DIAGNOSER:
                    current_aggregated_row_dict[diagnoser_name + AVERAGE_WASTED_EFFORT_NAME_SUFFIX] += test_result[diagnoser_name + WASTED_EFFORT_NAME_SUFFIX]
                    current_aggregated_row_dict[diagnoser_name + AVERAGE_FIX_ACCURACY_NAME_SUFFIX] += test_result[diagnoser_name + FIX_ACCURACY_NAME_SUFFIX]
                    current_aggregated_row_dict[diagnoser_name + AVERAGE_FIX_ACCURACY_INCREASE_NAME_SUFFIX] += test_result[diagnoser_name + FIX_ACCURACY_INCREASE_NAME_SUFFIX]
                raw_results = raw_results._append(test_result, ignore_index=True)
            if drifts_count == 0:
                continue
            current_aggregated_row_dict["drifts count"] = drifts_count
            for summarize_column_name in aggregated_summarizes_columns:
                current_aggregated_row_dict[summarize_column_name] /= drifts_count
            aggregated_results = aggregated_results._append(current_aggregated_row_dict, ignore_index=True)
        except Exception as e:
            if STOP_ON_EXCEPTION:
                raise e
            errors = errors._append({"name": dataset_name, "error": e}, ignore_index=True)
            continue

aggregating_total_row = {
    "name": "TOTAL",
    "tree size": raw_results["tree size"].mean(),
    "drifts count": aggregated_results["drifts count"].mean(),
    "average after accuracy decrease": raw_results["after accuracy decrease"].mean(),
    "average after retrain accuracy": raw_results["after retrain accuracy"].mean(),
    "average after retrain accuracy increase": raw_results["after retrain accuracy increase"].mean(),
    "average before after retrain accuracy": raw_results["before after retrain accuracy"].mean(),
    "average before after retrain accuracy increase": raw_results["before after retrain accuracy increase"].mean()
}
for diagnoser_name in DEFAULT_TESTING_DIAGNOSER:
    aggregating_total_row[diagnoser_name + AVERAGE_WASTED_EFFORT_NAME_SUFFIX] = raw_results[diagnoser_name + WASTED_EFFORT_NAME_SUFFIX].mean()
    aggregating_total_row[diagnoser_name + AVERAGE_FIX_ACCURACY_NAME_SUFFIX] = raw_results[diagnoser_name + FIX_ACCURACY_NAME_SUFFIX].mean()
    aggregating_total_row[diagnoser_name + AVERAGE_FIX_ACCURACY_INCREASE_NAME_SUFFIX] = raw_results[diagnoser_name + FIX_ACCURACY_INCREASE_NAME_SUFFIX].mean()

aggregated_results = aggregated_results._append(aggregating_total_row, ignore_index=True)

if not os.path.exists(RESULTS_FULL_PATH):
    os.mkdir(RESULTS_FULL_PATH)

aggregated_results.to_csv(f"{RESULTS_FILE_PATH_PREFIX}_aggregated.csv", index=False)
raw_results.to_csv(f"{RESULTS_FILE_PATH_PREFIX}_raw.csv", index=False)
if not errors.empty:
    errors.to_csv(f"{ERRORS_FILE_PATH_PREFIX}.csv", index=False)
elif os.path.exists(f"{ERRORS_FILE_PATH_PREFIX}.csv"):
    os.remove(f"{ERRORS_FILE_PATH_PREFIX}.csv")

print("All tests are done! average accuracy and the incremental:")
for diagnoser_name in DEFAULT_TESTING_DIAGNOSER:
    print(f"{diagnoser_name}: {aggregating_total_row[diagnoser_name + AVERAGE_FIX_ACCURACY_NAME_SUFFIX]}%, {aggregating_total_row[diagnoser_name + AVERAGE_FIX_ACCURACY_INCREASE_NAME_SUFFIX]}%")