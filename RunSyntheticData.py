import os

import xlsxwriter

from DataSet import DataSet
from SingleTree import run_single_tree_experiment
from HiddenPrints import HiddenPrints

import pandas as pd
import warnings
from datetime import datetime
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

directory = r"data\MOA"

feature_types = {
    "AgrawalGenerator" : ["numeric"]*3 + ["categorical"]*3 + ["numeric"]*3,
    "SEAGenerator" : ["numeric"]*3,
    "STAGGERGenerator" : ["categorical"]*3
}

def parse_file_name(f_name):
    parameters = f_name.split("_")
    generator = parameters[0]
    size = int(parameters[2])
    return generator, size

if __name__ == '__main__':
    all_results = []
    time_stamp = datetime.now()
    date_time = time_stamp.strftime("%d-%m-%Y__%H-%M-%S")

    all_datasets = []
    for filename in os.listdir(directory):
    # for filename in ["AgrawalGenerator_size_100_abrupt_peturbation_0_8to9.arff"]:
        print(f"################{filename}################")
        generator, size = parse_file_name(filename)
        path = os.path.join(directory, filename)
        dataset = DataSet(path, "synthetic", "class", feature_types[generator], size)

    # all_datasets = [
    #     DataSet("data/example3.arff", "synthetic", "class", ["numeric"]*3, 300),
    #     DataSet("data/example4.arff", "synthetic", "class", ["numeric"] * 3, 200)
    # ]
    #
    # for dataset in all_datasets:
    #     result = run_single_tree_experiment(dataset)
        with HiddenPrints():
            result = run_single_tree_experiment(dataset)
        result["dataset"] = dataset.name
        all_results.append(result)

    # write results to excel
    file_name = f"results/result_run_synthetic_MOA_{date_time}.xlsx"
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    # write headers
    dict_example = all_results[0]
    index_col = {}
    col_num = 0
    for key in dict_example.keys():
        worksheet.write(0, col_num, key)
        index_col[key] = col_num
        col_num += 1
    # write values
    row_num = 1
    for dict_res in all_results:
        for key, value in dict_res.items():
            if type(value) in (list, set, dict):
                value = str(value)
            col_num = index_col[key]
            worksheet.write(row_num, col_num, value)
        row_num += 1
    workbook.close()

    print("DONE")