from pandas import DataFrame, Series
from numpy import max as numpy_max
from typing import Generator
from copy import deepcopy

from APPETITE.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree
from APPETITE.Diagnosers import *

from APPETITE.Constants import DEFAULT_FIXING_DIAGNOSER

class Fixer:
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series,
                 diagnoser_name: str = DEFAULT_FIXING_DIAGNOSER,
                 *diagnoser_parameters: tuple[object]
    ):
        """
        Initialize the Fixer.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        diagnoser_name (str): The diagnoser name.
        diagnoser_parameters (tuple[object]): The diagnoser parameters.
        """
        self.mapped_tree = deepcopy(mapped_tree)
        self.feature_types = mapped_tree.data_feature_types
        self.X = X
        self.y = y
        self.diagnoser_name = diagnoser_name
        diagnoser_class, diagnoser_parameters = get_diagnoser(diagnoser_name, *diagnoser_parameters)
        self.diagnoser = diagnoser_class(self.mapped_tree, self.X, self.y, *diagnoser_parameters)
        self.faulty_nodes = None    # List of sk_indices of the faulty nodes; Lazy evaluation
        self.tree_already_fixed = False

    def _filter_data_reached_faults_generator(self,
                                  faults_count: int                           
        ) -> DataFrame | Generator[DataFrame, None, None]:
        """
        Filter the data that reached the faulty nodes.

        Parameters:
            faulty_nodes_count (int): The number of faulty nodes.

        Returns:
            DataFrame: The data that reached the faulty node if there is only one faulty node.
            Generator[DataFrame, None, None]: The data that reached each faulty node if there are more than one faulty node.
        """
        for faulty_node_index in self.faulty_nodes[: faults_count]:
            faulty_node = self.mapped_tree.get_node(faulty_node_index)
            filtered_data = faulty_node.get_data_reached_node(self.X)
            while filtered_data.empty and faulty_node.parent is not None:
                # Get the data that reached the parent node
                faulty_node = faulty_node.parent
                filtered_data = faulty_node.get_data_reached_node(self.X)
            yield filtered_data
    
    def _filter_data_reached_single_fault(self) -> DataFrame:
        """
        Filter the data that reached a single faulty node.

        Returns:
            DataFrame: The data that reached the faulty node.
        """
        return next(self._filter_data_reached_faults_generator(1))

    def _fix_terminal_faulty_node(self,
                                 faulty_node_index: int,
                                 data_reached_faulty_node: DataFrame
     ) -> None:
        """
        Fix a terminal faulty node.
        The fix is done by changing the class of the node to the most common class in the data that reached the node (after the drift).
        If the data that reached the node is empty, the fix is done by switching between the top and second top classes.
        
        Parameters:
            faulty_node_index (int): The index of the faulty node.
            data_reached_faulty_node (DataFrame): The data that reached the faulty node.
        """
        reached_labels = self.y[data_reached_faulty_node.index]
        
        values = self.mapped_tree.sklearn_tree_model.tree_.value[faulty_node_index]
        old_values = deepcopy(values)

        if len(reached_labels):
            # Get the most common class in the data that reached the faulty node
            most_common_class_index = reached_labels.value_counts().idxmax()

            # Make the most common class the class with the max count in the node
            max_value_count = numpy_max(values)
            values[0][most_common_class_index] = max_value_count + 1
        else:
            # Switch between top and second top classes
            max_value_index = values.argmax()
            max_value = values[0][max_value_index]
            values[0][max_value_index] = 0
            second_max_value_index = values.argmax()
            values[0][max_value_index] = max_value
            values[0][second_max_value_index] = max_value
        
        print(f"{self.diagnoser_name}: Faulty node {faulty_node_index} (Terminal) class changed from {old_values[0]} to {values[0]}")
        self.mapped_tree.sklearn_tree_model.tree_.value[faulty_node_index] = values


    def _fix_numeric_faulty_node(self, 
                                  faulty_node_index: int,
                                  data_reached_faulty_node: DataFrame
     ) -> None:
        """
        Fix a numeric faulty node.
        The fix is done by replacing the threshold of the node

        Parameters:
            faulty_node_index (int): The index of the faulty node.
            data_reached_faulty_node (DataFrame): The data that reached the faulty node.
        """
        faulty_node = self.mapped_tree.get_node(faulty_node_index)
        node_feature_average_before_drift = faulty_node.feature_average_value
        if node_feature_average_before_drift is None:
            raise NotImplementedError("The average feature value before the drift is not available")
        node_feature_average_after_drift = data_reached_faulty_node[faulty_node.feature].mean()
        node_feature_average_difference = node_feature_average_after_drift - node_feature_average_before_drift
        new_threshold = faulty_node.threshold + node_feature_average_difference
        print(f"{self.diagnoser_name}: Faulty node {faulty_node_index} (Numeric) threshold changed from {faulty_node.threshold} to {new_threshold}")
        self.mapped_tree.sklearn_tree_model.tree_.threshold[faulty_node_index] = new_threshold

    def _fix_categorical_faulty_node(self,
                                    faulty_node_index: int,
     ) -> None:
          """
          Fix a categorical faulty node.
          The fix is done by flipping the switch of the condition in it.
    
          Parameters:
                faulty_node_index (int): The index of the faulty node.
                data_reached_faulty_node (DataFrame): The data that reached the faulty node.
          """
          faulty_node = self.mapped_tree.get_node(faulty_node_index)
          left_child, right_child = faulty_node.left_child, faulty_node.right_child
          left_child_index, right_child_index = left_child.sk_index, right_child.sk_index
          sklearn_tree_model = self.mapped_tree.sklearn_tree_model
          sklearn_tree_model.tree_.children_left[faulty_node_index] = right_child_index
          sklearn_tree_model.tree_.children_right[faulty_node_index] = left_child_index
          print(f"{self.diagnoser_name}: Faulty node {faulty_node_index} (Categorical) condition flipped")
          
    def fix_faulty_node(self,
                        faulty_node_index: int,
                        data_reached_faulty_node: DataFrame
     ) -> None:
        """
        Fix a faulty node.

        Parameters:
            faulty_node_index (int): The index of the faulty node.
            data_reached_faulty_node (DataFrame): The data that reached the faulty node.
        """
        faulty_node = self.mapped_tree.get_node(faulty_node_index)
        if faulty_node.is_terminal():
            self._fix_terminal_faulty_node(faulty_node_index, data_reached_faulty_node)
            return
        faulty_node_feature_type = faulty_node.feature_type
        if faulty_node_feature_type is None:
            # Determine the type from the after drift dataset
            faulty_node_feature_type = self.feature_types[faulty_node.feature]
        if faulty_node_feature_type == "numeric":
            self._fix_numeric_faulty_node(faulty_node_index, data_reached_faulty_node)
        else:
            self._fix_categorical_faulty_node(faulty_node_index)
    
    def _create_fixed_mapped_tree(self) -> MappedDecisionTree:
        """
        Create new mapped decision tree after the fix.

        Returns:
            MappedDecisionTree: The fixed decision tree.
        """
        sklearn_tree_model = self.mapped_tree.sklearn_tree_model
        feature_types = self.mapped_tree.data_feature_types
        # Create a new MappedDecisionTree object with the fixed sklearn tree model
        fixed_mapped_decision_tree = MappedDecisionTree(sklearn_tree_model, feature_types=feature_types)
        self.mapped_tree = fixed_mapped_decision_tree
        self.tree_already_fixed = True
        return fixed_mapped_decision_tree

    def fix_single_fault(self, 
                         faulty_node: int = None
     ) -> tuple[MappedDecisionTree, int]:
        """
        Fix the decision tree under the assumption that there is a single faulty node in the tree.

        Parameters:
            faulty_node (int): The index of the faulty node. If None, the faulty node will be detected using the diagnoser.
        Returns:
            MappedDecisionTree: The fixed decision tree and the faulty node index.
        """
        if self.tree_already_fixed:
            return self.mapped_tree
        if faulty_node is None:
            self.faulty_nodes = self.diagnoser.get_diagnosis()
        else:
            self.faulty_nodes = [faulty_node]
        data_reached_faulty_node = self._filter_data_reached_single_fault()
        faulty_node_index = self.faulty_nodes[0]
        self.fix_faulty_node(faulty_node_index, data_reached_faulty_node)
        return self._create_fixed_mapped_tree(), faulty_node_index

    def fix_multiple_faults(self,
                            faulty_nodes: list[int]
                            ) -> MappedDecisionTree:
        """
        Fix all the faulty nodes in the decision tree.

        Parameters:
            faulty_nodes (list[int]): The indices of the faulty nodes.

        Returns:
            MappedDecisionTree: The fixed decision tree.
        """
        if self.tree_already_fixed:
            return self.mapped_tree
        if faulty_nodes is None:
            self.faulty_nodes = self.diagnoser.get_diagnosis()
        else:
            self.faulty_nodes = faulty_nodes
        for faulty_node_index, data_reached_faulty_node in zip(self.faulty_nodes, list(self._filter_data_reached_faults_generator(len(self.faulty_nodes)))):  
            self.fix_faulty_node(faulty_node_index, data_reached_faulty_node)
        return self._create_fixed_mapped_tree(), self.faulty_nodes