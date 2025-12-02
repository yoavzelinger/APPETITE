from copy import deepcopy

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from .MappedDecisionTree import MappedDecisionTree

class SubTreeReplaceableDecisionTree(DecisionTreeClassifier):
    """
    A Decision Tree Classifier that allows replacing subtrees.
    """
    def __init__(self, original_mapped_tree: MappedDecisionTree):
        self.mapped_tree = original_mapped_tree
        self.base_sklearn_tree_model = original_mapped_tree.sklearn_tree_model

        self.replaced_subtrees: dict[MappedDecisionTree.DecisionTreeNode, DecisionTreeClassifier] = {}

    def replace_subtree(self, new_node_index_to_replace: int, new_X: pd.DataFrame, new_y: pd.Series) -> None:
        """
        Replace a subtree rooted at the given node with a new subtree.

        Parameters:
            node_index_to_replace (int): The index of the node to replace.
            new_X (pd.DataFrame): The new input features for the subtree.
            new_y (pd.Series): The new target labels for the subtree.
        """
        assert new_node_index_to_replace in self.mapped_tree.tree_dict, "The specified node is not part of the tree."

        new_node_to_replace = self.mapped_tree.tree_dict[new_node_index_to_replace]

        for current_replaced_node in self.replaced_subtrees:
            if current_replaced_node.is_ancestor_of(new_node_to_replace):
                # The node to replace is part of an already replaced subtree
                return
            if current_replaced_node.is_successor_of(new_node_to_replace):
                # The node to replace is an ancestor of an already replaced subtree
                del self.replaced_subtrees[current_replaced_node]

        self.replaced_subtrees[new_node_to_replace] = deepcopy(self.base_sklearn_tree_model)
        self.replaced_subtrees[new_node_to_replace].fit(new_X, new_y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the class labels for the given input data.

        Parameters:
            X (pd.DataFrame): The input features.

        Returns:
            np.ndarray: The predicted class labels.
        """
        decision_paths = self.mapped_tree.sklearn_tree_model.decision_path(X)
        
        def get_prediction_tree(test_index: int) -> DecisionTreeClassifier:
            for replaced_node in self.replaced_subtrees:
                if decision_paths[test_index, replaced_node.sk_index]:
                    return self.replaced_subtrees[replaced_node]
            return self.base_sklearn_tree_model
        
        predictions = []
        for i in range(len(X)):
            prediction_tree = get_prediction_tree(i)
            predictions.append(prediction_tree.predict(X.iloc[[i]]))
        return np.array(predictions)