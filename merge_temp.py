print("Starting import")
from csv import DictReader
from pandas import DataFrame
from argparse import ArgumentParser
from datetime import datetime
from os import listdir, path as os_path, remove as os_remove

from Tester import tester_constants

parser = ArgumentParser(description="Run all tests")
parser.add_argument("-o", "--output", type=str, help="Output file name prefix, default is the result_TIMESTAMP", default=f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")
parser.add_argument("-c", "--clear", action="store_true", help="Clear the temporary files after merging, default is false", default=False)
args = parser.parse_args()

searching_directory = tester_constants.TEMP_RESULTS_FULL_PATH
searching_file_prefix = tester_constants.RESULTS_FILE_NAME_PREFIX

output_file_name = f"{tester_constants.RESULTS_FILE_NAME_PREFIX}_{args.output}.csv"

primary_key_columns = ["drift description", "after size"]
common_columns = ["drift size", "drifted features types", "tree size", "after accuracy decrease", "after retrain accuracy", "after retrain accuracy increase", "before after retrain accuracy", "before after retrain accuracy increase"]
diagnoser_columns = ["faulty features", "faulty nodes indicies", "fix accuracy", "fix accuracy increase"]

prefixes = ["", "fuzzy_"]
diagnosers = tester_constants.constants.DEFAULT_FIXING_DIAGNOSER
diagnoser_columns = ["faulty features", "faulty nodes indicies", "fix accuracy", "fix accuracy increase"]

diagnoser_index = 1
for prefix in prefixes:
    for diagnoser in diagnosers:
        print(diagnoser_index)
        diagnoser_index += 1
        if diagnoser_index == 2:
            continue
        current_searching_file_prefix = f"{searching_file_prefix}_{prefix}{diagnoser}"
        column_prefix = "fuzzy participation" if prefix == "fuzzy_" else ""
        current_diagnoser_columns = [f"{column_prefix} {diagnoser} {column}" for column in diagnoser_columns]
        current_output_df = DataFrame(columns=primary_key_columns + common_columns + current_diagnoser_columns)
        for file_index, file_name in enumerate(listdir(searching_directory), 1):
            # print("\t", file_index)
            if file_name.startswith(current_searching_file_prefix):
                print(f"Processing {file_index}: {file_name}")
                result_file_path = os_path.join(searching_directory, file_name)
                with open(result_file_path, "r") as file:
                    current_result_df = DataFrame(DictReader(file))
                    # check if the df has the index columns
                    if not all(column in current_result_df.columns for column in primary_key_columns):
                        print(f"Error while merging {file_name}, skipping")
                        continue
                    # _append the current result to the output df
                    current_output_df = current_output_df._append(current_result_df, ignore_index=True)
        # save
        if current_output_df.empty:
            print(f"temp folder do not contain any results for {prefix}{diagnoser}")
            continue
        # reorder columns
        ordered_columns = primary_key_columns + common_columns
        ordered_columns += [column for column in current_output_df.columns if column not in primary_key_columns and column not in common_columns and "fuzzy" not in column]
        ordered_columns += [column for column in current_output_df.columns if column not in ordered_columns]
        current_output_df = current_output_df[ordered_columns]
        # save the current output to a file
        print(f"Saving results for {prefix}{diagnoser}")
        output_full_path = os_path.join(tester_constants.TEMP_RESULTS_FULL_PATH, f"new_diagnoser_results_{current_searching_file_prefix}.csv")
        current_output_df.to_csv(output_full_path)
