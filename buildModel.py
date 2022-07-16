import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def build_model(data, features, target, model_type="tree", to_split=False):
    x_train = data[features]
    y_train = data[target]
    if to_split:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2,
                                                        random_state=1)  # 70% training and 30% test
    max_leaf_nodes = len(features) ** 2
    clf = DecisionTreeClassifier(min_samples_split=0.1, max_leaf_nodes=max_leaf_nodes)
    #clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    return clf
