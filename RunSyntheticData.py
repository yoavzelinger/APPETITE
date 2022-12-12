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

directory = r"data\MOA\gradual"

feature_types = {
    "AgrawalGenerator" : ["numeric"]*3 + ["categorical"]*3 + ["numeric"]*3,
    "SEAGenerator" : ["numeric"]*3,
    "STAGGERGenerator" : ["categorical"]*3
}

def parse_file_name(f_name):
    parameters = f_name.split("_")
    generator = parameters[0]
    size = int(parameters[2])
    window = int(parameters[4])
    if "noise" in parameters:
        noise  = int(parameters[7])
    elif "peturbation" in parameters:
        noise = int(float(parameters[7])*100)
    else: noise = -1
    return generator, size, window, noise

if __name__ == '__main__':
    all_results = []
    time_stamp = datetime.now()
    date_time = time_stamp.strftime("%d-%m-%Y__%H-%M-%S")
    samples_used = [0.1, 0.2, 0.3, 0.4]

    all_datasets = []
    for filename in os.listdir(directory):
        for n_used in samples_used:
            print(f"################{filename} - {n_used}################")
            generator, size, window, noise = parse_file_name(filename)
            path = os.path.join(directory, filename)
            dataset = DataSet(path, "synthetic", "class", feature_types[generator], [size, window, n_used, 0.6])

            with HiddenPrints():
                result = run_single_tree_experiment(dataset)

            result["dataset"] = dataset.name
            result["concept size"] = size
            result["generator"] = generator
            result["window"] = window
            result["noise"] = noise
            result["samples used"] = n_used
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