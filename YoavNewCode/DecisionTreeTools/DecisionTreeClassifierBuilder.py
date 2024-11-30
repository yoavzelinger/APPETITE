import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics

VALIDATION_SIZE = 0.2
DEFAULT_CROSS_VALIDATION_SPLIT_COUNT = 5
RANDOM_STATE = 7
NUMPY_RANDOM_STATE = 0
np.random.seed(NUMPY_RANDOM_STATE) # TODO - Check if needed

PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_leaf_nodes": [10, 20, 30]
}

def build(
        training_data: pd.DataFrame,
        features: list[str],
        target: str,
        validation_data: pd.DataFrame = None
        ) -> DecisionTreeClassifier:
    """
    Build a decision tree classifier based on the given data and features.

    Parameters:
        training_data (pd.DataFrame): The training data.
        features (list[str]): The features to use.
        target (str): The target column.
        validation_data (pd.DataFrame): The validation data. 
            If not provided, will use part of the training data.

    Returns:
        DecisionTreeClassifier: The decision tree classifier.
    """
    X_train = training_data[features]
    y_train = training_data[target]
    X_validation, y_validation = None, None
    if validation_data is not None:
        assert set(validation_data.columns) == set(training_data.columns), "Validation data must have the same columns as the training data"
        X_validation = validation_data[features]
        y_validation = validation_data[target]
    else:
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE)

    # Grid search modification
    modified_X_train, modified_y_train = X_train, y_train
    classes_counts = y_train.value_counts()
    if classes_counts.min() == 1:
        # Duplicate the rows with that one instance
        min_classes = classes_counts[classes_counts == 1].index
        for class_name in min_classes:
            sample_filter = y_train == class_name
            modified_X_train = modified_X_train.append(X_train[sample_filter, features], ignore_index=True)
            modified_y_train = modified_y_train.append(y_train[sample_filter], ignore_index=True)
    cross_validation_split_count = min(DEFAULT_CROSS_VALIDATION_SPLIT_COUNT, y_train.value_counts().min())

    decision_tree_classifier = DecisionTreeClassifier()
    # Find best parameters using grid search cross validation (on training data)
    grid_search_classifier = GridSearchCV(estimator=decision_tree_classifier, 
                                     param_grid=PARAM_GRID, 
                                     cv=cross_validation_split_count)
    grid_search_classifier.fit(modified_X_train, modified_y_train)
    grid_search_best_params = grid_search_classifier.best_params_ # Hyperparameters
    decision_tree_classifier = DecisionTreeClassifier(criterion=grid_search_best_params["criterion"], 
                                                      max_leaf_nodes=grid_search_best_params["max_leaf_nodes"])
    pruning_path = decision_tree_classifier.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = set(pruning_path.ccp_alphas) # TODO - Understand what is it
    best_decision_tree, best_accuracy = None, -1
    for ccp_alpha in ccp_alphas:
        if ccp_alpha < 0:
            continue
        current_decision_tree = DecisionTreeClassifier(criterion=grid_search_best_params["criterion"], 
                                                      max_leaf_nodes=grid_search_best_params["max_leaf_nodes"], 
                                                      ccp_alpha=ccp_alpha)
        current_decision_tree.fit(X_train, y_train)
        current_predictions = current_decision_tree.predict(X_validation)
        current_accuracy = metrics.accuracy_score(y_validation, current_predictions)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_decision_tree = current_decision_tree
            
    return best_decision_tree