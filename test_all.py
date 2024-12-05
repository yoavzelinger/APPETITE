import csv

from Tester.single_test_runner import run_test

DATA_DIRECTORY = "data"
DATASET_DESCRIPTION_FILE = "all_datasets.csv"
DATASETS_DIRECTORY = "Classification_Datasets"
DATASETS_FULL_PATH = f"{DATA_DIRECTORY}\\{DATASETS_DIRECTORY}\\"

# Get Description of all datasets
with open(f"{DATA_DIRECTORY}/{DATASET_DESCRIPTION_FILE}", "r") as descriptions_file, \
    open(f"{DATA_DIRECTORY}/aggregated_results.csv", "w") as aggregating_file, \
    open(f"{DATA_DIRECTORY}/raw_results.csv", "w") as results_file, \
        open(f"{DATA_DIRECTORY}/errors.csv", "w") as errors_file:
    aggregating_writer = csv.DictWriter(aggregating_file, fieldnames=["name", 
                                                                      "average accuracy drop", 
                                                                      "average fix accuracy increase"
                                                                      ])
    results_writer = csv.DictWriter(results_file, fieldnames=["drift description", 
                                                              "after accuracy decrease", 
                                                              "faulty node index", 
                                                              "faulty feature", 
                                                              "fix accuracy increase"])
    errors_writer = csv.DictWriter(errors_file, fieldnames=["name", "error"])
    
    aggregating_writer.writeheader()
    results_writer.writeheader()
    errors_writer.writeheader()
    descriptions_reader = csv.DictReader(descriptions_file)
    
    for dataset_description in descriptions_reader:
        dataset_name = dataset_description["name"]
        print(f"Running tests for {dataset_name}")
        drifts_count = 0
        total_after_accuracy_drop = 0
        total_fix_accuracy_increase = 0
        try:
            for test_result in run_test(DATASETS_FULL_PATH, dataset_name + ".csv"):
                drifts_count += 1
                total_after_accuracy_drop += test_result["after accuracy decrease"]
                total_fix_accuracy_increase += test_result["fix accuracy increase"]
                results_writer.writerow(test_result)
            aggregating_writer.writerow({"name": dataset_name, 
                                         "average accuracy drop": total_after_accuracy_drop / drifts_count, 
                                         "average fix accuracy increase": total_fix_accuracy_increase / drifts_count})
        except Exception as e:
            errors_writer.writerow({"name": dataset_name, "error": str(e)})
            continue
        if dataset_name == "ar4":
            break