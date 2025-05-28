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
            self.parent: 'MappedDecisionTree.DecisionTreeNode' = parent
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

        def get_feature_extended(self) -> str:
            """
            Get the feature extended name.
            If the node is a leaf then the parent's feature is returned.

            Returns:
                str: The extended feature .
            """
            return self.parent.feature if self.is_terminal() else self.feature
        
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
            if not self.is_terminal():
                # update the descendant stats
                self.left_child.update_node_data_attributes(X, y)
                self.right_child.update_node_data_attributes(X, y)
            if y is not None:
                # count correct classifications
                if self.is_terminal():
                    self.correct_classifications_count = (y == self.class_name).sum()
                    self.misclassifications_count = (y != self.class_name).sum()
                else:
                    self.correct_classifications_count = self.left_child.correct_classifications_count + self.right_child.correct_classifications_count
                    self.misclassifications_count = self.left_child.misclassifications_count + self.right_child.misclassifications_count
                if self.reached_samples_count > 0:
                    self.confidence = self.correct_classifications_count / self.reached_samples_count

        def __eq__(self, other):
            if isinstance(other, MappedDecisionTree.DecisionTreeNode):
                return self.sk_index == other.sk_index
            if isinstance(other, int):
                return self.sk_index == other
            return False

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
            self.root.update_node_data_attributes(X, y)
        self.spectra_dict = {node.spectra_index: node for node in self.tree_dict.values()}

    def map_tree(self, 
    ) -> dict[int, 'MappedDecisionTree.DecisionTreeNode']:
        self.sk_features = self.sklearn_tree_model.tree_.feature
        self.sk_thresholds = self.sklearn_tree_model.tree_.threshold
        self.sk_children_left = self.sklearn_tree_model.tree_.children_left
        self.sk_children_right = self.sklearn_tree_model.tree_.children_right
        self.sk_values = self.sklearn_tree_model.tree_.value
        sk_class_names = self.sklearn_tree_model.classes_
        
        tree_representation = {}
        nodes_to_check = [self.DecisionTreeNode(sk_index=0)]

        while len(nodes_to_check):
            current_node = nodes_to_check.pop(0)
            current_node.update_condition()
            current_index = current_node.sk_index
            tree_representation[current_index] = current_node
            left_child_index = self.sk_children_left[current_index]
            right_child_index = self.sk_children_right[current_index]

            if left_child_index == right_child_index:  # Leaf
                current_node_value = self.sk_values[current_index]
                class_name = argmax(current_node_value)
                class_name = sk_class_names[class_name]
                current_node.class_name = class_name
                continue
            
            current_node.threshold = self.sk_thresholds[current_index]
            feature_index = self.sk_features[current_index]
            current_node.feature = list(self.data_feature_types.keys())[feature_index]
            current_node.feature_type = self.data_feature_types[current_node.feature]
            right_child_index = self.sk_children_right[current_index]
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
    
    def prune_sibling_leaves(self,
                             leaf1: 'MappedDecisionTree.DecisionTreeNode',
                             leaf2: 'MappedDecisionTree.DecisionTreeNode'
     ) -> 'MappedDecisionTree.DecisionTreeNode':
        """
        Prune the sibling leaves.

        Parameters:
            leaf1 (DecisionTreeNode): The first leaf.
            leaf2 (DecisionTreeNode): The second leaf.

        Returns:
            DecisionTreeNode: The parent of the leaves.
        """
        self.tree_dict.pop(leaf1.sk_index)
        self.tree_dict.pop(leaf2.sk_index)
        parent = leaf1.parent
        parent.update_children(None, None)
        current_class = leaf1.class_name
        parent.feature, parent.feature_type, parent.threshold, parent.class_name = None, None, None, current_class
        # Adjust the tree
        parent_index = parent.sk_index
        self.sklearn_tree_model.tree_.children_left[parent_index] = TREE_LEAF
        self.sklearn_tree_model.tree_.children_right[parent_index] = TREE_LEAF
        self.sklearn_tree_model.tree_.feature[parent_index] = -2
        return parent
    
    def prune_leaf(self,
                   leaf_node: 'MappedDecisionTree.DecisionTreeNode'
     ) -> 'MappedDecisionTree.DecisionTreeNode':
        """
        Prune a leaf.
        The prune is done by removing the leaf and replacing the parent with it's sibling.
    
        Parameters:
            leaf (DecisionTreeNode): The leaf to prune.
        """
        parent, sibling = leaf_node.parent, leaf_node.get_sibling()
        self.tree_dict.pop(leaf_node.sk_index)
        self.tree_dict.pop(sibling.sk_index)
        # Replace all parent's attributes with siblings'
        parent.update_children(sibling.left_child, sibling.right_child)
        parent.feature, parent.feature_type, parent.threshold, parent.class_name = sibling.feature, sibling.feature_type, sibling.threshold, sibling.class_name
        # Update the sklearn tree
        parent_index = parent.sk_index
        self.sk_children_left[parent_index] = sibling.left_child.sk_index if not sibling.is_terminal() else TREE_LEAF
        self.sk_children_right[parent_index] = sibling.right_child.sk_index if not sibling.is_terminal() else TREE_LEAF
        self.sk_features[parent_index] = self.sk_features[sibling.sk_index]
        self.sk_thresholds[parent_index] = sibling.threshold
        self.sk_values[parent_index] = self.sk_values[sibling.sk_index]
        if parent.is_terminal():
            return parent
    
    def prune_tree(self) -> None:
        leaf_nodes = [node for node in self.tree_dict.values() if node.is_terminal()]
        tree_changed = False
        while len(leaf_nodes):
            current_leaf = leaf_nodes.pop(0)
            if current_leaf.sk_index not in self.tree_dict: # Already pruned
                continue
            sibling = current_leaf.get_sibling()
            if sibling is None: # Root
                continue
            if sibling.is_terminal() and current_leaf.class_name == sibling.class_name: # Sibling is a leaf with the same class
                # leaf_nodes = [leaf_node for leaf_node in leaf_nodes if leaf_node.sk_index != sibling.sk_index] # Remove sibling from the list
                if sibling in leaf_nodes:
                    leaf_nodes.remove(sibling)
                leaf_nodes += [self.prune_sibling_leaves(current_leaf, sibling)]
                tree_changed = True
            elif hasattr(current_leaf, "reached_samples_count") and not current_leaf.reached_samples_count: # Redundant leaf
                new_leaf = self.prune_leaf(current_leaf)
                tree_changed = True
                if new_leaf:
                    leaf_nodes += [new_leaf]
            
        if tree_changed: # Attributes changed
            self.update_tree_attributes()

    def __repr__(self) -> str:
        return export_text(self.sklearn_tree_model, feature_names=list(self.data_feature_types.keys()))
