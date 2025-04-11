from csv import DictReader
from pandas import DataFrame
from argparse import ArgumentParser
from datetime import datetime
from os import listdir, path as os_path, remove as os_remove

from Tester import *

parser = ArgumentParser(description="Run all tests")
parser.add_argument("-o", "--output", type=str, help="Output file name prefix, default is the result_TIMESTAMP", default=f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")
# add flag -d to deterime if the user wants to clear the temporary files
parser.add_argument("-c", "--clear", action="store_true", help="Clear the temporary files after merging, default is false", default=False)
args = parser.parse_args()


searching_directory = TEMP_RESULTS_FULL_PATH
searching_file_prefix = RESULTS_FILE_NAME_PREFIX

output_file_name = f"{RESULTS_FILE_NAME_PREFIX}_{args.output}.csv"

# merge all files in the directory to a single file
output_df = DataFrame()

for file_name in listdir(searching_directory):
    if file_name.startswith(searching_file_prefix):
        result_file_path = os_path.join(searching_directory, file_name)
        with open(result_file_path, "r") as file:
            result_file_reader = DictReader(file)
            result_df = DataFrame(result_file_reader)
            output_df = DataFrame._append(output_df, result_df, ignore_index=True)
        if args.clear:  # remove the file after merging
            print(f"Removing {result_file_path}")
            os_remove(result_file_path)

if output_df.empty:
    print("temp folder do not contain any results")
else:
    output_full_path = os_path.join(RESULTS_FULL_PATH, output_file_name)
    output_df.to_csv(output_full_path, index=False)
    print(f"Results merged to {output_full_path}")