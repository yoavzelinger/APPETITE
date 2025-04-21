print("Starting import")
from csv import DictReader
from pandas import DataFrame
from argparse import ArgumentParser
from datetime import datetime
from os import listdir, path as os_path
from copy import deepcopy as copy

from Tester import tester_constants

parser = ArgumentParser(description="Run all tests")
parser.add_argument("-o", "--output", type=str, help="Output file name prefix, default is the result_TIMESTAMP", default=f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")
args = parser.parse_args()

searching_directory = tester_constants.TEMP_RESULTS_FULL_PATH
searching_file_prefix = tester_constants.RESULTS_FILE_NAME_PREFIX
searching_fuzzy_file_prefix = tester_constants.RESULTS_FUZZY_PARTICIPATION_FILE_NAME_PREFIX

output_file_name = f"{tester_constants.RESULTS_FILE_NAME_PREFIX}_{args.output}.csv"
fuzzy_output_file_name = f"{tester_constants.RESULTS_FUZZY_PARTICIPATION_FILE_NAME_PREFIX}_{args.output}.csv"

group_by_columns = ["drift description", "after size", "drift size", "drifted features types"]
common_aggregated_columns = ["after accuracy decrease", "after retrain accuracy increase", "before after retrain accuracy increase"]
diagnoser_aggregated_columns_suffixes = ["fix accuracy increase", "wasted effort"]

aggregated_columns, fuzzy_aggregated_columns = copy(common_aggregated_columns), copy(common_aggregated_columns)
aggregating_functions_dict = {common_aggregated_column: "mean" for common_aggregated_column in common_aggregated_columns}
fuzzy_aggregating_functions_dict = copy(aggregating_functions_dict)
aggregating_functions_dict["drift description"], fuzzy_aggregating_functions_dict["drift description"] = "count", "count"
for diagnoser_name in tester_constants.constants.DEFAULT_FIXING_DIAGNOSER:
    for diagnoser_aggregated_column_suffix in diagnoser_aggregated_columns_suffixes:
        diagnoser_aggregated_column = f"{diagnoser_name} {diagnoser_aggregated_column_suffix}"
        aggregated_columns.append(diagnoser_aggregated_column)
        fuzzy_aggregated_columns.append(f"fuzzy participation {diagnoser_aggregated_column}")
        aggregating_functions_dict[diagnoser_aggregated_column], fuzzy_aggregating_functions_dict[diagnoser_aggregated_column] = "mean", "mean"


output_df, fuzzy_output_df = DataFrame(columns=group_by_columns + aggregated_columns), DataFrame(columns=group_by_columns + fuzzy_aggregated_columns)

for current_file_index, current_file_name in enumerate(listdir(tester_constants.TEMP_RESULTS_FULL_PATH), 1):
    print("Working on file", current_file_index)
    if not current_file_name.startswith(tester_constants.RESULTS_FILE_NAME_PREFIX):
        continue
    relevant_output_df, relevant_aggregating_functions_dict = output_df, aggregating_functions_dict
    if current_file_name.startswith(tester_constants.RESULTS_FUZZY_PARTICIPATION_FILE_NAME_PREFIX):
        relevant_output_df, relevant_aggregating_functions_dict = fuzzy_output_df, fuzzy_aggregating_functions_dict
    
    with open(os_path.join(tester_constants.TEMP_RESULTS_FULL_PATH, current_file_name), "r") as current_file:
        current_results_df = DataFrame(DictReader(current_file))
        current_group_by_df = current_results_df.groupby(group_by_columns).agg(relevant_aggregating_functions_dict).reset_index()
        relevant_output_df = relevant_output_df._append(current_group_by_df, ignore_index=True)

print(f"Saving results")
output_full_path, fuzzy_output_full_path = os_path.join(tester_constants.RESULTS_FULL_PATH, f"{output_file_name}.csv"), os_path.join(tester_constants.RESULTS_FULL_PATH, f"{output_file_name}.csv")
output_df.to_csv(output_full_path)
fuzzy_output_df.to_csv(fuzzy_output_full_path)