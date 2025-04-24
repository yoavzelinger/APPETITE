print("Starting import")
from pandas import DataFrame, read_csv
from argparse import ArgumentParser
from datetime import datetime
from os import listdir, path as os_path
from copy import deepcopy as copy

from Tester import tester_constants

parser = ArgumentParser(description="Run all tests")
parser.add_argument("-o", "--output", type=str, help="Output file name prefix, default is the result_TIMESTAMP", default=f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")
args = parser.parse_args()

group_by_columns = ["after size", "drift size", "drifted features types"]
common_aggregated_columns = ["after accuracy decrease", "after retrain accuracy increase", "before after retrain accuracy increase"]
diagnoser_aggregated_columns_suffixes = ["fix accuracy increase", "wasted effort"]

columns_dtypes = {
    "drift description": "string",
    "after size": "float64",
    "drift size": "int64",
    "drifted features types": "string",
    "after accuracy decrease": "float64",
    "after retrain accuracy increase": "float64",
    "before after retrain accuracy increase": "float64",
    "fix accuracy increase": "float64",
    "wasted effort": "int64"
}

fuzzy_columns_dtypes = copy(columns_dtypes)

aggregated_columns, fuzzy_aggregated_columns = copy(common_aggregated_columns), copy(common_aggregated_columns)
aggregating_functions_dict = {common_aggregated_column: "sum" for common_aggregated_column in common_aggregated_columns}
fuzzy_aggregating_functions_dict = copy(aggregating_functions_dict)
for diagnoser_name in tester_constants.constants.DEFAULT_FIXING_DIAGNOSER:
    for diagnoser_aggregated_column_suffix in diagnoser_aggregated_columns_suffixes:
        diagnoser_aggregated_column = f"{diagnoser_name} {diagnoser_aggregated_column_suffix}"
        fuzzy_diagnoser_aggregated_column = f"fuzzy participation {diagnoser_aggregated_column}"
        aggregated_columns.append(diagnoser_aggregated_column)
        fuzzy_aggregated_columns.append(fuzzy_diagnoser_aggregated_column)
        aggregating_functions_dict[diagnoser_aggregated_column], fuzzy_aggregating_functions_dict[fuzzy_diagnoser_aggregated_column] = "sum", "sum"
        columns_dtypes[diagnoser_aggregated_column], fuzzy_columns_dtypes[fuzzy_diagnoser_aggregated_column] = "float64", "float64"
count_column_name = "drift description"
aggregating_functions_dict[count_column_name], fuzzy_aggregating_functions_dict[count_column_name] = "count", "count"


output_dfs = [DataFrame(columns=group_by_columns + aggregated_columns + ["count"]), DataFrame(columns=group_by_columns + fuzzy_aggregated_columns + ["count"])]

for current_file_index, current_file_name in enumerate(listdir(tester_constants.TEMP_RESULTS_FULL_PATH), 1):
    print("Working on file", current_file_index, ":", current_file_name)
    if not current_file_name.startswith(tester_constants.RESULTS_FILE_NAME_PREFIX):
        continue
    relevant_output_df_index, relevant_aggregating_functions_dict, relevant_columns_dtypes = 0, aggregating_functions_dict, columns_dtypes
    if current_file_name.startswith(tester_constants.RESULTS_FUZZY_PARTICIPATION_FILE_NAME_PREFIX):
        relevant_output_df_index, relevant_aggregating_functions_dict, relevant_columns_dtypes = 1, fuzzy_aggregating_functions_dict, fuzzy_columns_dtypes
    
    with open(os_path.join(tester_constants.TEMP_RESULTS_FULL_PATH, current_file_name), "r") as current_file:
        current_results_df = read_csv(current_file, dtype=relevant_columns_dtypes)
        current_group_by_df = current_results_df.groupby(group_by_columns).agg(relevant_aggregating_functions_dict).reset_index()
        # rename count_column_name to count
        current_group_by_df.rename(columns={count_column_name: "count"}, inplace=True)
        output_dfs[relevant_output_df_index] = output_dfs[relevant_output_df_index]._append(current_group_by_df, ignore_index=True)

print(f"Saving results")
# move count column to be after the group by columns
for output_df, aggregated_columns, result_file_name_prefix in zip(output_dfs, [aggregated_columns, fuzzy_aggregated_columns], [tester_constants.RESULTS_FILE_NAME_PREFIX, tester_constants.RESULTS_FUZZY_PARTICIPATION_FILE_NAME_PREFIX]):
    output_df = output_df[group_by_columns + ["count"] + aggregated_columns]
    output_full_path = os_path.join(tester_constants.RESULTS_FULL_PATH, f"{result_file_name_prefix}_{args.output}.csv")
    output_df.to_csv(output_full_path, index=False)

# make group by columns the keys
output_dfs[0].set_index(group_by_columns, inplace=True)
output_dfs[1].set_index(group_by_columns, inplace=True)

# combine the two dataframes using combine_first
output_df = output_dfs[0].combine_first(output_dfs[1])
output_df.reset_index(inplace=True)
output_df = output_df[group_by_columns + ["count"] + aggregated_columns + fuzzy_aggregated_columns]
output_full_path = os_path.join(tester_constants.RESULTS_FULL_PATH, f"combined_{args.output}.csv")
output_df.to_csv(output_full_path, index=False)
print(f"Results saved to {output_full_path}")