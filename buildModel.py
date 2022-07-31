import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from DataSet import DataSet

param_grid_tree = {
    "criterion": ["gini", "entropy"],
    # "min_samples_split": [2, 0.1, 0.05, 3],
    "max_depth": [4, 6, 8, 10, 12],
    # "min_samples_leaf": [1, 2, 5, 10],
    "max_leaf_nodes": [10, 20, 30]
}


def build_model(data, features, target, model_type="tree", to_split=False, val_data=None):
    x_train_all = data[features]
    y_train_all = data[target]

    if to_split:
        if len(x_train_all) > 1:
            x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.2, random_state=7)  # 80% training and 20% test
        else:
            x_train, x_val = x_train_all, x_train_all
            y_train, y_val = y_train_all, y_train_all
    else:
        x_val = val_data[features]
        y_val = val_data[target]
        x_train, y_train = x_train_all, y_train_all

    x_train1 = x_train
    y_train1 = y_train
    all_y = y_train.unique()
    y_count = y_train.value_counts()
    min_y_count = y_count.min()
    if min_y_count == 1:
        only_1 = np.where(y_count == 1)
        for y_loc in only_1:  # add another sample to each class
            sample_filter = y_train == all_y[y_loc[0]]
            x_train1 = x_train1.append(x_train.loc[sample_filter,features], ignore_index=True)
            y_train1 = y_train1.append(y_train[sample_filter], ignore_index=True)
        min_y_count = 2
    n_split = min(5, min_y_count)

    dec_tree = DecisionTreeClassifier()
    clf_GS = GridSearchCV(estimator=dec_tree, param_grid=param_grid_tree, cv=n_split)
    clf_GS.fit(x_train1, y_train1)
    best_params = clf_GS.best_params_

    # train tree on best params and then prune
    clf = DecisionTreeClassifier(criterion=best_params["criterion"], max_depth=best_params["max_depth"],
                                     max_leaf_nodes=best_params["max_leaf_nodes"])
    path = clf.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas = set(path.ccp_alphas)
    clfs = []
    accuracy_val = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(criterion=best_params["criterion"], max_depth=best_params["max_depth"],
                                     max_leaf_nodes=best_params["max_leaf_nodes"], ccp_alpha=ccp_alpha)
        clf.fit(x_train, y_train)
        clfs.append(clf)
        pred = clf.predict(x_val)
        accuracy = metrics.accuracy_score(y_val, pred)
        accuracy_val.append(accuracy)

    best = np.argmax(np.array(accuracy_val))
    best_clf = clfs[best]
    return best_clf

def map_tree(tree):
    class_names = tree.classes_
    tree_representation = {0: {"depth": 0,
                "parent": -1}}
    nodes_to_check = [0]
    while len(nodes_to_check) > 0:
        node = nodes_to_check.pop(0)

        left_child = tree.tree_.children_left[node]
        tree_representation[node]["left"] = left_child
        if left_child != -1:
            tree_representation[left_child] = {"parent":node,
                                               "type": "left"}
            nodes_to_check.append(left_child)
        right_child = tree.tree_.children_right[node]
        tree_representation[node]["right"] = right_child
        if right_child != -1:
            tree_representation[right_child] = {"parent":node,
                                               "type": "right"}
            nodes_to_check.append(right_child)

        tree_representation[node]["feature"] = tree.tree_.feature[node]
        tree_representation[node]["threshold"] = tree.tree_.threshold[node]

        if node != 0:
            parent = tree_representation[node]["parent"]
            tree_representation[node]["depth"] = tree_representation[parent]["depth"] + 1
            parent_cond = tree_representation[parent]["condition"]
            sign = "<=" if tree_representation[node]["type"] == "left" else ">"
            #cond = f"{model.tree_.feature[parent]} {sign} {model.tree_.threshold[parent]}"
            cond = {
                "feature": tree.tree_.feature[parent],
                "sign": sign,
                "thresh": tree.tree_.threshold[parent]
            }
            tree_representation[node]["condition"] = parent_cond + [cond]
        else:  # root
            tree_representation[node]["condition"] = []

        if left_child == -1:  # leaf
            value = tree.tree_.value[node]
            class_name = np.argmax(value)
            class_name = class_names[class_name]
            tree_representation[node]["class"] = class_name

    return tree_representation

def prune_tree(tree, tree_rep):
    node_list = list(range(tree.tree_.node_count))
    non_leaf_nodes = list(filter(lambda n: tree_rep[n]["left"] != -1, node_list))
    leaf_nodes = list(filter(lambda n: tree_rep[n]["left"] == -1, node_list))
    for leaf in leaf_nodes:
        parent = tree_rep[leaf]["parent"]
        if tree_rep[leaf]["type"] == "left":
            brother = tree_rep[parent]["right"]
        else:
            brother = tree_rep[parent]["left"]
        if brother in leaf_nodes and brother["class"] == leaf["class"]:  # prune
            leaf_nodes.remove(brother)
            leaf_nodes.remove(leaf)
            # todo: change
            leaf_nodes.append(parent)
            non_leaf_nodes.remove(parent)

    pass

def print_tree_rules(tree, feature_names):
    tree_rules = export_text(tree, feature_names=feature_names)
    print(tree_rules)


if __name__ == '__main__':
    sizes = (0.75, 0.05, 0.2)
    all_datasets_build = [
        DataSet("data/real/winequality-white.csv", "diagnosis_check", "quality", ["numeric"]*11, sizes, name="winequality-white", to_shuffle=True),
        DataSet("data/real/abalone.data", "diagnosis_check", "rings", ["categorical"] + ["numeric"]*7,  sizes, name="abalone", to_shuffle=True),
        DataSet("data/real/data_banknote_authentication.txt", "diagnosis_check", "class", ["numeric"]*4, sizes, name="data_banknote_authentication", to_shuffle=True),
        DataSet("data/real/pima-indians-diabetes.csv", "diagnosis_check", "class", ["numeric"]*8, sizes, name="pima-indians-diabetes", to_shuffle=True)
]
    for dataset in all_datasets_build:
        concept_size = dataset.before_size
        train = dataset.data.iloc[0:int(0.9 * concept_size)]
        validation = dataset.data.iloc[int(0.9 * concept_size):concept_size]
        model = build_model(train, dataset.features, dataset.target, val_data=validation)
        map_tree(model)
        print("TREE:")
        print_tree_rules(model, dataset.features)
        print(f"number of nodes: {model.tree_.node_count}")
