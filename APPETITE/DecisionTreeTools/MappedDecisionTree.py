from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.tree._tree import TREE_LEAF
from pandas import DataFrame, Series
from numpy import argmax

class MappedDecisionTree:
    class DecisionTreeNode:
        def __init__(self, 
                     sk_index: int = 0,
                     parent = None, 
                     left_child = None, 
                     right_child = None,
                     feature: str | None = None,
                     feature_type: str | None = None,
                     threshold: float | None = None,
                     class_name: str | None = None,
                     spectra_index: int = -1
        ):
            """
            Initialize the DecisionTreeNode.

            Parameters:
                sk_index (int): The index of the node (in the sklearn tree).
                parent (DecisionTreeNode): The parent node.
                left_child (DecisionTreeNode): The left child node.
                right_child (DecisionTreeNode): The right child node.
                feature (str): The feature.
                threshold (float): The threshold.
                class_name (str): The class name.
                spectra_index (int): The index of the node (in the spectra matrix).
            """
            self.sk_index, self.spectra_index = sk_index, spectra_index
            self.parent = parent
            self.update_children(left_child, right_child)
            self.feature = feature
            self.feature_type = feature_type
            self.threshold = threshold
            self.class_name = class_name
            self.conditions_path = []
            self.depth = 0 if parent is None else parent.depth + 1
            self.feature_average_value = None

        def update_children(self, 
                            left_child: 'MappedDecisionTree.DecisionTreeNode', 
                            right_child: 'MappedDecisionTree.DecisionTreeNode'
        ) -> None:
            """
            Update the children of the node.

            Parameters:
                left_child (DecisionTreeNode): The left child node.
                right_child (DecisionTreeNode): The right child node.
            """
            # A node can be either a terminal node (two children) or a non-terminal node (a leaf - no children)
            assert (left_child is None) == (right_child is None)
            self.left_child = left_child
            self.right_child = right_child

        def update_condition(self) -> None:
            """
            Update the conditions path of the node.
            
            The conditions path is the path from the root to the node. Each condition is a dictionary with the following items:
                "feature" (str): The feature.
                "sign" (str): The sign of the threshold.
                "threshold" (float): The threshold.
            """
            if self.parent is None:
                return
            current_condition = {
                "feature": self.parent.feature,
                "sign": "<=" if self.is_left_child() else ">",
                "threshold": self.parent.threshold
            }
            self.conditions_path = self.parent.conditions_path + [current_condition]

        def is_terminal(self) -> bool:
            """
            Check if the node is a terminal node.
            
            Returns:
                bool: True if the node is a terminal node, False otherwise.
            """
            return self.left_child is None
        
        def is_left_child(self) -> bool:
            """
            Check if the node is a left child.
            
            Returns:
                bool: True if the node is a left child, False otherwise."""
            if self.parent is None:
                return False
            return self.parent.left_child == self
        
        def is_right_child(self) -> bool:
            """
            Check if the node is a right child.
            
            Returns:
                bool: True if the node is a right child, False otherwise.
            """
            if self.parent is None:
                return False
            return not self.is_left_child()
        
        def get_sibling(self) -> 'MappedDecisionTree.DecisionTreeNode':
            """
            Get the sibling of the node.
            
            Returns:
                DecisionTreeNode: The sibling of the node.
            """
            if self.parent is None:
                return None
            return self.parent.left_child if self.is_right_child() else self.parent.right_child
        
        def __repr__(self) -> str:
            """
            Get the string representation of the node.
            """
            return str(self.sk_index)
        
        def get_data_reached_node(self,
                                  X: DataFrame,
                                  y: Series = None
         ) -> DataFrame | tuple[DataFrame, Series]:
            """
            Filter the data that reached the node.

            Parameters:
                X (DataFrame): The data.
                y (Series): The target column.

            Returns:
                DataFrame | tuple[DataFrame, Series]: The data that reached the node.
            """
            for condition in self.conditions_path:
                feature, sign, threshold = condition.values()
                assert feature in X.columns, f"Feature {feature} not in the dataset"
                if sign == "<=":
                    X = X[X[feature] <= threshold]
                else:
                    X = X[X[feature] > threshold]
                if y is not None:
                    y = y[X.index]
            return X if y is None else (X, y)
        
        def update_node_data_attributes(self, 
                                        X: DataFrame,
                                        y: Series = None
         ) -> None:
            """
            Update the average feature value of the node.
            The average is calculated on the data that reached the node.

            Parameters:
                X (DataFrame): The data.
                y (Series): The target column
            """
            # Get the data that reached the node
            X = self.get_data_reached_node(X, y)
            if y is not None:
                X, y = X
            self.reached_samples_count = len(X)
            if self.feature_type == "numeric":
                self.feature_average_value = X[self.feature].mean()
            if self.is_terminal() and y is not None:
                # count correct classifications
                self.correct_classifications_count = (y == self.class_name).sum()
                self.misclassifications_count = (y != self.class_name).sum()

    def __init__(self, 
                 sklearn_tree_model: DecisionTreeClassifier,
                 feature_types: dict[str, str] = None,
                 prune: bool = True,
                 X: DataFrame = None,
                 y: Series = None
    ):
        """
        Initialize the MappedDecisionTree.
        
        Parameters:
            sklearn_tree_model (DecisionTreeClassifier): The sklearn decision tree.
            prune (bool): Whether to prune the tree.
            X (DataFrame): The data.
            y (Series): The target column.
        """
        assert sklearn_tree_model is not None and feature_types is not None
        self.sklearn_tree_model = sklearn_tree_model
        self.data_feature_types = feature_types

        self.criterion = sklearn_tree_model.criterion # TODO - make sure that needed
        

        self.tree_dict = self.map_tree()
        self.root = self.get_node(0)
        self.update_tree_attributes(X, y)
        if prune:
            self.prune_tree()
        self.update_tree_attributes(X, y)

    def update_tree_attributes(self,
                               X: DataFrame = None,
                               y: Series = None
     ) -> None:
        """
        Update the tree attributes. those attributes are aggregated from the nodes.

        Parameters:
            X (DataFrame): The data. If provided - calculating the data reached each node and the average feature value.
            y (Series): The target column. If provided - calculating the correct classifications count.
        """
        self.node_count = len(self.tree_dict)
        self.max_depth = max(map(lambda node: node.depth, self.tree_dict.values()))
        self.tree_features_set, self.classes_set = set(), set()
        for ordered_index, node in enumerate(self.tree_dict.values()):
            node.update_condition()
            node.spectra_index = ordered_index
            if node.feature:
                self.tree_features_set.add(node.feature)
            if node.class_name:
                self.classes_set.add(node.class_name)
            if X is not None:
                node.update_node_data_attributes(X, y)
        self.spectra_dict = {node.spectra_index: node for node in self.tree_dict.values()}

    def map_tree(self, 
    ) -> dict[int, 'MappedDecisionTree.DecisionTreeNode']:
        sk_features = self.sklearn_tree_model.tree_.feature
        sk_thresholds = self.sklearn_tree_model.tree_.threshold
        sk_children_left = self.sklearn_tree_model.tree_.children_left
        sk_children_right = self.sklearn_tree_model.tree_.children_right
        sk_values = self.sklearn_tree_model.tree_.value
        sk_class_names = self.sklearn_tree_model.classes_
        
        tree_representation = {}
        nodes_to_check = [self.DecisionTreeNode(sk_index=0)]

        while len(nodes_to_check):
            current_node = nodes_to_check.pop(0)
            current_node.update_condition()
            current_index = current_node.sk_index
            tree_representation[current_index] = current_node
            left_child_index = sk_children_left[current_index]
            right_child_index = sk_children_right[current_index]

            if left_child_index == right_child_index:  # Leaf
                current_node_value = sk_values[current_index]
                class_name = argmax(current_node_value)
                class_name = sk_class_names[class_name]
                current_node.class_name = class_name
                continue
            
            current_node.threshold = sk_thresholds[current_index]
            feature_index = sk_features[current_index]
            current_node.feature = list(self.data_feature_types.keys())[feature_index]
            current_node.feature_type = self.data_feature_types[current_node.feature]
            right_child_index = sk_children_right[current_index]
            for child_index in (left_child_index, right_child_index):
                child_node = self.DecisionTreeNode(sk_index=child_index, parent=current_node)
                nodes_to_check.append(child_node)
            current_node.update_children(*(nodes_to_check[-2: ]))

        return tree_representation
    
    def get_node(self, 
                 index: int,
                 use_spectra_index: bool = False
     ) -> 'MappedDecisionTree.DecisionTreeNode':
        if use_spectra_index:
            return self.spectra_dict[index]
        return self.tree_dict[index]
    
    def convert_node_index_to_spectra_index(self,
                                            index: int
    ) -> int:
        return self.get_node(index).spectra_index
    
    def convert_spectra_index_to_node_index(self,
                                            index: int
    ) -> int:
        return self.spectra_dict[index].sk_index
    
    def prune_tree(self) -> None:
        leaf_nodes = [node for node in self.tree_dict.values() if node.is_terminal()]
        pruned_indices = []
        while len(leaf_nodes):
            current_leaf = leaf_nodes.pop(0)
            sibling = current_leaf.get_sibling()

            if sibling is None: # Root
                continue
            if not sibling.is_terminal():
                continue
            if current_leaf.class_name != sibling.class_name:
                continue
            # Prune
            pruned_indices += [current_leaf.sk_index, sibling.sk_index]
            self.tree_dict.pop(current_leaf.sk_index)
            self.tree_dict.pop(sibling.sk_index)
            # Make parent a leaf
            parent = current_leaf.parent
            parent.update_children(None, None)
            current_class = current_leaf.class_name
            parent.feature, parent.feature_type, parent.threshold, parent.class_name = None, None, None, current_class
            leaf_nodes += [parent]
            # Adjust the tree
            parent_index = parent.sk_index
            self.sklearn_tree_model.tree_.children_left[parent_index] = TREE_LEAF
            self.sklearn_tree_model.tree_.children_right[parent_index] = TREE_LEAF
            self.sklearn_tree_model.tree_.feature[parent_index] = -2
        if len(pruned_indices): # Attributes changed
            self.update_tree_attributes()
            print(f"Pruned {len(pruned_indices)} nodes from the tree. Pruned nodes: {pruned_indices}")

    def __repr__(self) -> str:
        return export_text(self.sklearn_tree_model, feature_names=list(self.data_feature_types.keys()))
