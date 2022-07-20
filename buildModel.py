import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import decomposition, datasets
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from DataSet import DataSet
from updateModel import print_tree_rules

param_grid_tree = {
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 0.1, 0.05, 3],
    "max_depth": [4, 6, 8, 10, 12],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_leaf_nodes": [10, 20, 30]
}


def build_model(data, features, target, model_type="tree", to_split=False):
    x_train = data[features]
    y_train = data[target]
    if to_split:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2,
                                                        random_state=7)  # 80% training and 20% test

    dec_tree = tree.DecisionTreeClassifier()
    clf_GS = GridSearchCV(estimator=dec_tree, param_grid=param_grid_tree)
    clf_GS.fit(x_train, y_train)
    clf = clf_GS.best_estimator_
    print(f'best_params_: {clf_GS.best_params_}')
    print(f'best_score_: {clf_GS.best_score_}')

    # max_leaf_nodes = len(features) ** 2
    # clf = DecisionTreeClassifier(min_samples_split=0.1, max_leaf_nodes=max_leaf_nodes)
    # #clf = DecisionTreeClassifier()
    # clf = clf.fit(x_train, y_train)
    return clf


if __name__ == '__main__':
    dataset = DataSet("data/real/winequality-white.csv", "diagnosis_check", "quality", 2000, ["numeric"] * 11,
            name="winequality-white", to_shuffle=True)
    concept_size = dataset.batch_size
    model = build_model(dataset.data.iloc[0:int(0.9 * concept_size)], dataset.features, dataset.target)
    print("TREE:")
    print_tree_rules(model, dataset.features)
    print(f"number of nodes: {model.tree_.node_count}")

