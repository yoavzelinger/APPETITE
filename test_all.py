import os
from csv import DictReader
from pandas import DataFrame

from Tester.single_test_runner import run_test

DATA_DIRECTORY = "data"
DATASET_DESCRIPTION_FILE = "all_datasets.csv"
DATASETS_DIRECTORY = "Classification_Datasets"
DATASETS_FULL_PATH = f"{DATA_DIRECTORY}\\{DATASETS_DIRECTORY}\\"
RESULTS_DIRECTORY = "results"
RESULTS_FULL_PATH = f"{DATA_DIRECTORY}\\{RESULTS_DIRECTORY}\\"

# Create DataFrame for the aggregated results
aggregated_results = DataFrame(columns=["name", "tree size", "drifts count", "average accuracy drop precentage", "average fix accuracy increase precentage"])
raw_results = DataFrame(columns=["drift description", "tree size", "after accuracy decrease precentage", "faulty node index", "faulty feature", "fix accuracy increase precentage"])
errors = DataFrame(columns=["name", "error"])

with open(f"{DATA_DIRECTORY}/{DATASET_DESCRIPTION_FILE}", "r") as descriptions_file:
    descriptions_reader = DictReader(descriptions_file)
    for dataset_description in descriptions_reader:
        dataset_name = dataset_description["name"]
        print(f"Running tests for {dataset_name}")
        drifts_count = 0
        tree_size = - -1
        total_after_accuracy_drop = 0
        total_fix_accuracy_increase = 0
        try:
            for test_result in run_test(DATASETS_FULL_PATH, dataset_name + ".csv"):
                drifts_count += 1
                tree_size = test_result["tree size"]
                total_after_accuracy_drop += test_result["after accuracy decrease precentage"]
                total_fix_accuracy_increase += test_result["fix accuracy increase precentage"]
                raw_results = raw_results._append(test_result, ignore_index=True)
            if drifts_count == 0:
                continue
            aggregated_results = aggregated_results._append({"name": dataset_name, 
                                                            "tree size": tree_size if tree_size != -1 else "N/A",
                                                            "drifts count": drifts_count,
                                                            "average accuracy drop precentage": total_after_accuracy_drop / drifts_count if drifts_count != 0 else "N/A", 
                                                            "average fix accuracy increase precentage": total_fix_accuracy_increase / drifts_count if drifts_count != 0 else "N/A"
                                                            }, ignore_index=True)
        except Exception as e:
            errors = errors._append({"name": dataset_name, "error": str(e)}, ignore_index=True)
            continue

average_tree_size = raw_results["tree size"].mean()
average_drifts_count = aggregated_results["drifts count"].mean()
after_accuracy_drop_precentage = raw_results["after accuracy decrease precentage"].mean()
fix_accuracy_increase_precentage = raw_results["fix accuracy increase precentage"].mean()

aggregated_results = aggregated_results._append({"name": "TOTAL", 
                                                "tree size": average_tree_size,
                                                "drifts count": average_drifts_count,
                                                "average accuracy drop precentage": after_accuracy_drop_precentage,
                                                "average fix accuracy increase precentage": fix_accuracy_increase_precentage
                                                }, ignore_index=True)
raw_results = raw_results._append({"drift description": "TOTAL", 
                                    "tree size": average_tree_size,
                                    "after accuracy decrease precentage": after_accuracy_drop_precentage,
                                    "faulty node index": "N/A",
                                    "faulty feature": "N/A",
                                    "fix accuracy increase precentage": fix_accuracy_increase_precentage
                                    }, ignore_index=True)

if not os.path.exists(RESULTS_FULL_PATH):
    os.mkdir(RESULTS_FULL_PATH)
aggregated_results.to_csv(f"{RESULTS_FULL_PATH}/aggregated_results.csv", index=False)
raw_results.to_csv(f"{RESULTS_FULL_PATH}/all_results.csv", index=False)
if not errors.empty:
    errors.to_csv(f"{RESULTS_FULL_PATH}/errors.csv", index=False)

print("All tests are done!, average accuracy increased by", fix_accuracy_increase_precentage, "%")