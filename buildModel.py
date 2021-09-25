import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split


def build_model(data, features, target, model_type="tree"):
    x = data[features]
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=1)  # 70% training and 30% test
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    return clf
