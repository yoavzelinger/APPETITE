print("Starting import")
from csv import DictReader
from pandas import DataFrame
from argparse import ArgumentParser
from datetime import datetime
from os import listdir, path as os_path, remove as os_remove

from Tester import tester_constants

parser = ArgumentParser(description="Run all tests")
parser.add_argument("-o", "--output", type=str, help="Output file name prefix, default is the result_TIMESTAMP", default=f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")
args = parser.parse_args()

searching_directory = tester_constants.TEMP_RESULTS_FULL_PATH
searching_file_prefix = "new_diagnoser_results_results_"

output_file_name = f"{tester_constants.RESULTS_FILE_NAME_PREFIX}_{args.output}.csv"

primary_key_columns = ["drift description", "after size"]
common_columns = ["drift size", "drifted features types", "tree size", "after accuracy decrease", "after retrain accuracy", "after retrain accuracy increase", "before after retrain accuracy", "before after retrain accuracy increase"]
diagnoser_columns = ["faulty features", "faulty nodes indicies", "fix accuracy", "fix accuracy increase"]

output_df = DataFrame(columns=primary_key_columns + common_columns + diagnoser_columns).set_index(primary_key_columns)

# go over the files in the directory and find the ones that start with the searching_file_prefix
file_names = [f for f in listdir(searching_directory) if os_path.isfile(os_path.join(searching_directory, f)) and f.startswith(searching_file_prefix)]
for file_index, file_name in enumerate(file_names):
    if file_index > 1:
        break
    file_path = os_path.join(searching_directory, file_name)
    with open(file_path, "r") as file:
        # open df
        current_df = DataFrame(DictReader(file)).set_index(primary_key_columns)
        output_df = output_df.combine_first(current_df)

# save
output_df.to_csv(os_path.join(tester_constants.RESULTS_DIRECTORY, output_file_name))output_df.to_csv(os_path.join(tester_constants.RESULTS_FULL_PATH, output_file_name))output_df.to_csv(os_path.join(tester_constants.RESULTS_FULL_PATH, output_file_name))