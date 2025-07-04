import os
from sys import exit
from argparse import ArgumentParser
from datetime import datetime
from csv import DictReader
from pandas import DataFrame


from Tester import *

from warnings import simplefilter as warnings_simplefilter
warnings_simplefilter(action='ignore', category=FutureWarning)

DIAGNOSES_NAME_SUFFIX = " diagnoses"
FAULTY_FEATURES_NAME_SUFFIX = " faulty features"
WASTED_EFFORT_NAME_SUFFIX = " wasted effort"
CORRECTLY_IDENTIFIED_NAME_SUFFIX = " correctly_identified"
AVERAGE_WASTED_EFFORT_NAME_SUFFIX = " average" + WASTED_EFFORT_NAME_SUFFIX
AVERAGE_CORRECTLY_IDENTIFIED_NAME_SUFFIX = " average" + CORRECTLY_IDENTIFIED_NAME_SUFFIX
FIX_ACCURACY_NAME_SUFFIX = " fix accuracy"
AVERAGE_FIX_ACCURACY_NAME_SUFFIX = " average" + FIX_ACCURACY_NAME_SUFFIX
FIX_ACCURACY_INCREASE_NAME_SUFFIX = " fix accuracy increase"
AVERAGE_FIX_ACCURACY_INCREASE_NAME_SUFFIX = " average" + FIX_ACCURACY_INCREASE_NAME_SUFFIX

parser = ArgumentParser(description="Run all tests")
parser.add_argument("-o", "--output", type=str, help="Output file name prefix, default is the result_TIMESTAMP", default=f"{datetime.now().strftime('%d-%m_%H-%M-%S')}")
parser.add_argument("-s", "--skip", action="store_true", help=f"skip exceptions, default is {tester_constants.SKIP_EXCEPTIONS}. If true will write the errors to errors file", default=tester_constants.SKIP_EXCEPTIONS)
parser.add_argument("-n", "--names", type=str, nargs="+", help="Specific datasets to run, default is all", default=[])
parser.add_argument("-p", "--prefixes", type=str, nargs="+", help="prefixes to datasets to run, default is all", default=[])
parser.add_argument("-a", "--after_window", type=float, nargs="+", help="After window sizes, default is all", default=tester_constants.AFTER_WINDOW_TEST_SIZES)
parser.add_argument("-d", "--drift_size", type=int, help=f"size of the drift, default is between {tester_constants.MIN_DRIFT_SIZE} and {tester_constants.MAX_DRIFT_SIZE}", default=-1)
parser.add_argument("-c", "--count", type=int, help="Number of tests to run, default is running all", default=-1)
parser.add_argument("-t", "--test", type=str, help="Test dataset to run if you want to run a specific test")

args = parser.parse_args()
# print(f"Running config: {args}")
diagnosers_data = load_testing_diagnosers_data()
diagnosers_output_names = list(map(lambda diagnoser_data: diagnoser_data["output_name"], diagnosers_data))
after_windows_string = ""
if args.after_window != tester_constants.AFTER_WINDOW_TEST_SIZES:
    tester_constants.AFTER_WINDOW_TEST_SIZES = args.after_window
    after_windows_string = "-".join(map(str, tester_constants.AFTER_WINDOW_TEST_SIZES))
if args.drift_size > 0:
    tester_constants.MIN_DRIFT_SIZE, tester_constants.MAX_DRIFT_SIZE = args.drift_size, args.drift_size
print(f"Running tests with {len(diagnosers_output_names)} diagnosers: {diagnosers_output_names}")
skip_exceptions = args.skip
datasets_count = args.count
if args.test:
    single_test.sanity_run(file_name=args.test + ".csv", diagnosers_data=diagnosers_data)
    exit(0)
SPECIFIC_DATASETS = args.names
if SPECIFIC_DATASETS:
    print(f"Running tests with prefixes: {SPECIFIC_DATASETS}")
    SPECIFIC_DATASETS = list(map(lambda prefix: prefix.lower(), SPECIFIC_DATASETS))

raw_results_columns = ["dataset name","after size", "drift size", "drift description", "drifted features", "drifted features types", "total drift type", "drift severity level", "tree size", "tree features count", "after accuracy decrease", "after retrain accuracy", "after retrain accuracy increase", "before after retrain accuracy", "before after retrain accuracy increase"]
aggregated_groupby_columns = ["dataset name", "tree size", "drifts count"]

aggregated_summarizes_columns = ["average after accuracy decrease", "average after retrain accuracy", "average after retrain accuracy increase", "average before after retrain accuracy", "average before after retrain accuracy increase"]
for diagnoser_output_name in diagnosers_output_names:
    raw_results_columns.append(diagnoser_output_name + FAULTY_FEATURES_NAME_SUFFIX)
    raw_results_columns.append(diagnoser_output_name + DIAGNOSES_NAME_SUFFIX)
    raw_results_columns.append(diagnoser_output_name + WASTED_EFFORT_NAME_SUFFIX)
    raw_results_columns.append(diagnoser_output_name + CORRECTLY_IDENTIFIED_NAME_SUFFIX)
    raw_results_columns.append(diagnoser_output_name + FIX_ACCURACY_NAME_SUFFIX)
    raw_results_columns.append(diagnoser_output_name + FIX_ACCURACY_INCREASE_NAME_SUFFIX)
    aggregated_summarizes_columns.append(diagnoser_output_name + AVERAGE_WASTED_EFFORT_NAME_SUFFIX)
    aggregated_summarizes_columns.append(diagnoser_output_name + AVERAGE_CORRECTLY_IDENTIFIED_NAME_SUFFIX)
    aggregated_summarizes_columns.append(diagnoser_output_name + AVERAGE_FIX_ACCURACY_NAME_SUFFIX)
    aggregated_summarizes_columns.append(diagnoser_output_name + AVERAGE_FIX_ACCURACY_INCREASE_NAME_SUFFIX)

# Create DataFrame for the aggregated results
raw_results = DataFrame(columns=raw_results_columns)
aggregated_results = DataFrame(columns=aggregated_groupby_columns + aggregated_summarizes_columns)
errors = DataFrame(columns=["name", "error"])

with open(tester_constants.DATASET_DESCRIPTION_FILE_PATH, "r") as descriptions_file:
    descriptions_reader = DictReader(descriptions_file)
    for dataset_description in descriptions_reader:
        if not datasets_count:
            break
        dataset_name = dataset_description["name"]
        if SPECIFIC_DATASETS and not any(map(lambda specific_dataset: dataset_name.lower() == specific_dataset, SPECIFIC_DATASETS)):
            continue
        datasets_count -= 1
        drifts_count = 0

        current_aggregated_row_dict = {
            "dataset name": dataset_name,
            "tree size": -1,
            "average after accuracy decrease": 0,
            "average after retrain accuracy": 0,
            "average after retrain accuracy increase": 0,
            "average before after retrain accuracy": 0,
            "average before after retrain accuracy increase": 0
        }
        current_aggregated_row_dict.update({summarize_column_name: 0 for summarize_column_name in aggregated_summarizes_columns})
        print(f"Running tests for {dataset_name}")
        for test_result in run_single_test(tester_constants.DATASETS_FULL_PATH, dataset_name + ".csv", diagnosers_data=diagnosers_data):
            if isinstance(test_result, Exception):
                if not skip_exceptions:
                    raise test_result
                errors = errors._append({"name": dataset_name, "error": test_result}, ignore_index=True)
                continue
            test_result["dataset name"] = dataset_name
            drifts_count += 1
            current_aggregated_row_dict["tree size"] = test_result["tree size"]
            current_aggregated_row_dict["average after accuracy decrease"] += test_result["after accuracy decrease"]
            current_aggregated_row_dict["average after retrain accuracy"] += test_result["after retrain accuracy"]
            current_aggregated_row_dict["average after retrain accuracy increase"] += test_result["after retrain accuracy increase"]
            current_aggregated_row_dict["average before after retrain accuracy"] += test_result["before after retrain accuracy"]
            current_aggregated_row_dict["average before after retrain accuracy increase"] += test_result["before after retrain accuracy increase"]
            for diagnoser_output_name in diagnosers_output_names:
                current_aggregated_row_dict[diagnoser_output_name + AVERAGE_WASTED_EFFORT_NAME_SUFFIX] += test_result[diagnoser_output_name + WASTED_EFFORT_NAME_SUFFIX]
                current_aggregated_row_dict[diagnoser_output_name + AVERAGE_CORRECTLY_IDENTIFIED_NAME_SUFFIX] += test_result[diagnoser_output_name + CORRECTLY_IDENTIFIED_NAME_SUFFIX]
                current_aggregated_row_dict[diagnoser_output_name + AVERAGE_FIX_ACCURACY_NAME_SUFFIX] += test_result[diagnoser_output_name + FIX_ACCURACY_NAME_SUFFIX]
                current_aggregated_row_dict[diagnoser_output_name + AVERAGE_FIX_ACCURACY_INCREASE_NAME_SUFFIX] += test_result[diagnoser_output_name + FIX_ACCURACY_INCREASE_NAME_SUFFIX]
            raw_results = raw_results._append(test_result, ignore_index=True)
        if drifts_count == 0:
            continue
        current_aggregated_row_dict["drifts count"] = drifts_count
        for summarize_column_name in aggregated_summarizes_columns:
            current_aggregated_row_dict[summarize_column_name] /= drifts_count
        aggregated_results = aggregated_results._append(current_aggregated_row_dict, ignore_index=True)

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
for diagnoser_output_name in diagnosers_output_names:
    aggregating_total_row[diagnoser_output_name + AVERAGE_WASTED_EFFORT_NAME_SUFFIX] = raw_results[diagnoser_output_name + WASTED_EFFORT_NAME_SUFFIX].mean()
    aggregating_total_row[diagnoser_output_name + AVERAGE_CORRECTLY_IDENTIFIED_NAME_SUFFIX] = raw_results[diagnoser_output_name + CORRECTLY_IDENTIFIED_NAME_SUFFIX].mean()
    aggregating_total_row[diagnoser_output_name + AVERAGE_FIX_ACCURACY_NAME_SUFFIX] = raw_results[diagnoser_output_name + FIX_ACCURACY_NAME_SUFFIX].mean()
    aggregating_total_row[diagnoser_output_name + AVERAGE_FIX_ACCURACY_INCREASE_NAME_SUFFIX] = raw_results[diagnoser_output_name + FIX_ACCURACY_INCREASE_NAME_SUFFIX].mean()

aggregated_results = aggregated_results._append(aggregating_total_row, ignore_index=True)

RESULTS_FULL_PATH = tester_constants.TEMP_RESULTS_FULL_PATH if SPECIFIC_DATASETS else tester_constants.RESULTS_FULL_PATH
os.makedirs(RESULTS_FULL_PATH, exist_ok=True)

RESULTS_FILE_PATH_PREFIX = os_path.join(RESULTS_FULL_PATH, tester_constants.RESULTS_FILE_NAME_PREFIX)
ERRORS_FILE_PATH_PREFIX = os_path.join(RESULTS_FULL_PATH, tester_constants.ERRORS_FILE_NAME_PREFIX)
if args.drift_size > 0:
    RESULTS_FILE_PATH_PREFIX += f"_drift_size_{args.drift_size}"
    ERRORS_FILE_PATH_PREFIX += f"_drift_size_{args.drift_size}"
if after_windows_string:
    RESULTS_FILE_PATH_PREFIX += f"_after_window_{after_windows_string}"
    ERRORS_FILE_PATH_PREFIX += f"_after_window_{after_windows_string}"

file_name_suffix = "-".join(SPECIFIC_DATASETS) if SPECIFIC_DATASETS else args.output
RESULTS_FILE_PATH_PREFIX, ERRORS_FILE_PATH_PREFIX = f"{RESULTS_FILE_PATH_PREFIX}_{file_name_suffix}", f"{ERRORS_FILE_PATH_PREFIX}_{file_name_suffix}"

if not SPECIFIC_DATASETS:
    aggregated_results.to_csv(f"{RESULTS_FILE_PATH_PREFIX}_aggregated.csv", index=False)
if not raw_results.empty:
    raw_results.to_csv(f"{RESULTS_FILE_PATH_PREFIX}.csv", index=False)

if not errors.empty:
    errors.to_csv(f"{ERRORS_FILE_PATH_PREFIX}.csv", index=False)
elif os.path.exists(f"{ERRORS_FILE_PATH_PREFIX}.csv"):
    os.remove(f"{ERRORS_FILE_PATH_PREFIX}.csv")

print("All tests are done! average accuracy and the incremental:")
for diagnoser_output_name in diagnosers_output_names:
    print(f"{diagnoser_output_name}: {aggregating_total_row[diagnoser_output_name + AVERAGE_FIX_ACCURACY_NAME_SUFFIX]}%, {aggregating_total_row[diagnoser_output_name + AVERAGE_FIX_ACCURACY_INCREASE_NAME_SUFFIX]}%")