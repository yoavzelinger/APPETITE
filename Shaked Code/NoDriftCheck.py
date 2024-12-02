import xlsxwriter

from DataSet import DataSet
from SingleTree import run_single_tree_experiment
from HiddenPrints import HiddenPrints

import pandas as pd
import warnings
from datetime import datetime
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

if __name__ == '__main__':
    all_results = []
    time_stamp = datetime.now()
    date_time = time_stamp.strftime("%d-%m-%Y__%H-%M-%S")

    all_sizes = [
        (0.7, 0.1, 0.2),
        (0.7, 0.07, 0.2),
        (0.7, 0.05, 0.2),
        (0.7, 0.02, 0.2)
    ]

    all_datasets = pd.read_csv('data/all_datasets.csv', index_col=0)

    for sizes in all_sizes:
        for index, row in all_datasets.iterrows():
            # if index > 3:
            #     break
            dataset = DataSet(row["path"].replace("\\", "/"), "no_drift", None, None, sizes, name=row["name"], to_shuffle=True)
            try:
                with HiddenPrints():
                    result = run_single_tree_experiment(dataset)
                result["size"] = sizes[1]
                result["dataset"] = dataset.name
                all_results.append(result)

            except Exception as e:
                print(f"failed in run: {dataset.name.upper()} {sizes}")
                print(e)

    # write results to excel
    file_name = f"results/result_run_NO_DRIFT_{date_time}.xlsx"
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
            try:
                worksheet.write(row_num, col_num, value)
            except TypeError:
                print(f"{dataset.name.upper()} {sizes} - problem with key: '{key}', value: {value}")
        row_num += 1
    workbook.close()

    print("DONE")