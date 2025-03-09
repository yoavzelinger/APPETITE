from sys import argv as sys_argv
from pandas import concat as pd_concat
from sklearn.metrics import accuracy_score

from APPETITE import *

from Tester.Constants import *
import traceback

def get_dataset(directory: str,
                file_name: str,
                proportions_tuple: int | tuple[float] = PROPORTIONS_TUPLE,
                after_window_size: float = AFTER_WINDOW_SIZE
                )-> Dataset:
    return Dataset(directory + file_name, proportions_tuple, after_window_size)

def get_sklearn_tree(X_train,
                     y_train):
    return build_tree(X_train, y_train)

def get_mapped_tree(sklearn_tree_model, feature_types, X_train, y_train):
    return MappedDecisionTree(sklearn_tree_model, feature_types=feature_types, X=X_train, y=y_train)

def drift_single_tree_feature(mapped_tree: MappedDecisionTree, 
                              dataset: Dataset):
    """
    Generate a drifted in a single tree that is used in the tree structure.
    """
    tree_features_set = mapped_tree.tree_features_set
    for drifting_feature in tree_features_set:
        after_drift_generator = dataset.drift_generator(drifting_feature, partition="after")
        test_drift_generator = dataset.drift_generator(drifting_feature, partition="test")
        for ((X_after_drifted, y_after), after_drift_description, drifted_features), ((X_test_drifted, y_test), _, _) in zip(after_drift_generator, test_drift_generator):
            yield (X_after_drifted, y_after,), (X_test_drifted, y_test), after_drift_description[len("after") + 1: ], drifted_features[0]

def drift_tree(mapped_tree: MappedDecisionTree,
               dataset: Dataset
               ):
    assert CURRENT_DRIFT_TYPE in SUPPORTED_DRIFT_TYPES, f"{CURRENT_DRIFT_TYPE} drift type is not supported. Supported drift types: {SUPPORTED_DRIFT_TYPES}"
    if CURRENT_DRIFT_TYPE == SINGLE_DRIFT_TYPE:
        yield from drift_single_tree_feature(mapped_tree, dataset)
    else:
        raise NotImplementedError(f"{CURRENT_DRIFT_TYPE} drift type is not implemented.")

def get_wasted_effort(mapped_tree: MappedDecisionTree,
                               diagnosed_faulty_nodes_indices: list[int],
                               true_faulty_features: set[str],
                               require_full_fix = True
 ) -> int:
    """
    Calculate the wasted effort of the diagnoser.
    In here the wasted effort is calculated for fixes of all the nodes that include a faulty feature.

    Parameters:
    mapped_tree (MappedDecisionTree): The mapped decision tree.
    diagnosed_faulty_nodes_indices (list[int]): The indices of the diagnosed faulty nodes.
    true_faulty_features (list[str]): The true faulty features.

    Returns:
    int: The wasted effort.
    """
    get_node_feature_func = lambda node: node.feature if node.feature is not None else node.parent.feature
    # Get node's feature or it's parent's feature if it's None
    # If not require_full_fix, then the faulty feature will be counted only as one
    faulty_features_nodes_counts = {true_faulty_feature : int(not require_full_fix) for true_faulty_feature in true_faulty_features}
    for tree_node in mapped_tree.tree_dict.values():
        tree_node_feature = get_node_feature_func(tree_node)
        if tree_node_feature in true_faulty_features:
            if require_full_fix:
                faulty_features_nodes_counts[tree_node_feature] += 1

    wasted_effort = 0
    for diagnosed_faulty_node in map(mapped_tree.get_node, diagnosed_faulty_nodes_indices):
        diagnosed_faulty_feature = get_node_feature_func(diagnosed_faulty_node)
        if diagnosed_faulty_feature in true_faulty_features:
            faulty_features_nodes_counts[diagnosed_faulty_feature] -= 1
            if not any(faulty_features_nodes_counts.values()):
                break
        else:
            wasted_effort += 1
    return wasted_effort

def get_accuracy(model, X, y):
    y_predicted = model.predict(X)
    return accuracy_score(y, y_predicted)

def run_single_test(directory, file_name, proportions_tuple=PROPORTIONS_TUPLE, after_window_size=AFTER_WINDOW_SIZE, diagnoser_names=DEFAULT_TESTING_DIAGNOSER, *diagnoser_parameters):
    dataset = get_dataset(directory, file_name, proportions_tuple, after_window_size)

    X_train, y_train = dataset.get_before_concept()
    sklearn_tree_model = get_sklearn_tree(X_train, y_train)

    X_test, y_test = dataset.get_test_concept()
    no_drift_test_accuracy = get_accuracy(sklearn_tree_model, X_test, y_test)
    if no_drift_test_accuracy < MINIMUM_ORIGINAL_ACCURACY:  # Original model is not good enough
        return

    X_after_original, y_after_original = dataset.get_after_concept()
    no_drift_after_accuracy = get_accuracy(sklearn_tree_model, X_after_original, y_after_original)

    mapped_tree = get_mapped_tree(sklearn_tree_model, dataset.feature_types, X_train, y_train)

    for (X_after_drifted, y_after), (X_test_drifted, y_test), drift_description, drifted_feature in drift_tree(mapped_tree, dataset):
        try:
            after_accuracy = get_accuracy(mapped_tree.sklearn_tree_model, X_after_drifted, y_after) # Original model
            after_accuracy_drop = no_drift_after_accuracy - after_accuracy

            test_accuracy = get_accuracy(mapped_tree.sklearn_tree_model, X_test_drifted, y_test) # Original model
            test_accuracy_drop = no_drift_test_accuracy - test_accuracy
            if after_accuracy_drop < MINIMUM_DRIFT_ACCURACY_DROP or test_accuracy_drop < MINIMUM_DRIFT_ACCURACY_DROP:   # insignificant drift
                continue

            after_retrained_tree = get_sklearn_tree(X_after_drifted, y_after)
            after_retrained_accuracy = get_accuracy(after_retrained_tree, X_test_drifted, y_test)
            after_retrained_accuracy_bump = after_retrained_accuracy - test_accuracy

            X_before_after_concat, y_before_after_concat = pd_concat([X_train, X_after_drifted]), pd_concat([y_train, y_after])
            before_after_retrained_tree = get_sklearn_tree(X_before_after_concat, y_before_after_concat)
            before_after_retrained_accuracy = get_accuracy(before_after_retrained_tree, X_test_drifted, y_test)
            before_after_retrained_accuracy_bump = before_after_retrained_accuracy - test_accuracy

            current_results_dict = {
                "drift description": drift_description,
                "tree size": mapped_tree.node_count,
                "after accuracy decrease": after_accuracy_drop * 100,
                "after retrain accuracy": after_retrained_accuracy * 100,
                "after retrain accuracy increase": after_retrained_accuracy_bump * 100,
                "before after retrain accuracy": before_after_retrained_accuracy * 100,
                "before after retrain accuracy increase": before_after_retrained_accuracy_bump * 100
            }
            if isinstance(diagnoser_names, str):
                diagnoser_names = (diagnoser_names, )
            for diagnoser_name in diagnoser_names:
                fixer = Fixer(mapped_tree, X_after_drifted, y_after, diagnoser_name=diagnoser_name, *diagnoser_parameters)
                fixed_mapped_tree, faulty_nodes_indicies = fixer.fix_tree()
                faulty_features = [mapped_tree.get_node(faulty_node_index).feature for faulty_node_index in faulty_nodes_indicies]
                fixed_test_accuracy = get_accuracy(fixed_mapped_tree.sklearn_tree_model, X_test_drifted, y_test)
                test_accuracy_bump = fixed_test_accuracy - test_accuracy
                drifted_feature = drifted_feature if isinstance(drifted_feature, set) else set([drifted_feature])
                wasted_effort = get_wasted_effort(mapped_tree, fixer.faulty_nodes, drifted_feature, WASTED_EFFORT_REQUIRE_FULL_FIX)
                current_results_dict.update({
                    f"{diagnoser_name} faulty nodes indicies": ", ".join(map(str, faulty_nodes_indicies)),
                    f"{diagnoser_name} faulty features": ", ".join(str(faulty_features)),
                    f"{diagnoser_name} wasted effort": wasted_effort,
                    f"{diagnoser_name} fix accuracy": fixed_test_accuracy * 100,
                    f"{diagnoser_name} fix accuracy increase": test_accuracy_bump * 100
                })
            yield current_results_dict
        except Exception as e:
            exception_class = e.__class__.__name__
            tb = traceback.extract_tb(e.__traceback__)
            error_line, error_file = tb[-1].lineno, tb[-1].filename
            raise Exception(f"{exception_class} in {drift_description} ({error_file}, line {error_line}): {e}")
        
def get_example_mapped_tree(directory=DATASETS_FULL_PATH, file_name=EXAMPLE_FILE_NAME):
    dataset = get_dataset(directory, file_name + ".csv")
    X_train, y_train = dataset.get_before_concept()
    sklearn_tree_model = get_sklearn_tree(X_train, y_train)
    return get_mapped_tree(sklearn_tree_model, dataset.feature_types, X_train, y_train)

def sanity_run(directory=DATASETS_FULL_PATH, file_name=EXAMPLE_FILE_NAME):
    for result in run_single_test(directory, file_name):
        print(result)
        
if __name__ == "__main__":
    file_name = EXAMPLE_FILE_NAME if len(sys_argv) < 2 else sys_argv[1]
    sanity_run(DATASETS_FULL_PATH, file_name + ".csv")