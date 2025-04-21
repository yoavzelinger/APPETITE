from csv import DictReader
from pandas import DataFrame
from argparse import ArgumentParser
from datetime import datetime
from os import listdir, path as os_path, remove as os_remove

from Tester import *

parser = ArgumentParser(description="Run all tests")
parser.add_argument("-o", "--output", type=str, help="Output file name prefix, default is the result_TIMESTAMP", default=f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")
parser.add_argument("-c", "--clear", action="store_true", help="Clear the temporary files after merging, default is false", default=False)
args = parser.parse_args()


searching_directory = tester_constants.TEMP_RESULTS_FULL_PATH
searching_file_prefix = tester_constants.RESULTS_FILE_NAME_PREFIX

output_file_name = f"{tester_constants.RESULTS_FILE_NAME_PREFIX}_{args.output}.csv"

primary_key_columns = ["drift description", "after size"]
common_columns = ["drift size", "drifted features types", "tree size", "after accuracy decrease", "after retrain accuracy", "after retrain accuracy increase", "before after retrain accuracy", "before after retrain accuracy increase"]
output_df = DataFrame(columns=primary_key_columns + common_columns).set_index(primary_key_columns)

for file_name in listdir(searching_directory):
    if file_name.startswith(searching_file_prefix):
        print(f"Processing {file_name}")
        result_file_path = os_path.join(searching_directory, file_name)
        with open(result_file_path, "r") as file:
            result_df = DataFrame(DictReader(file))
            try:
                result_df = result_df.set_index(primary_key_columns)
                output_df = output_df.combine_first(result_df)
            except:
                print(f"Error while merging {file_name}, skipping")
                continue
        if args.clear:
            print(f"Removing {result_file_path}")
            os_remove(result_file_path)

# reorder columns
ordered_columns = common_columns
ordered_columns += [column for column in output_df.columns if column not in primary_key_columns and column not in common_columns and "fuzzy" not in column] # regulars
ordered_columns += [column for column in output_df.columns if column not in ordered_columns] # fuzzy
output_df = output_df[ordered_columns]

if output_df.empty:
    print("temp folder do not contain any results")
else:
    output_full_path = os_path.join(tester_constants.RESULTS_FULL_PATH, output_file_name)
    output_df.to_csv(output_full_path)
    print(f"Results merged to {output_full_path}")