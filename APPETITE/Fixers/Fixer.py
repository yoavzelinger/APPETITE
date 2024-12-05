from pandas import DataFrame, Series
from pandas.api.types import is_numeric_dtype
from numpy import max as numpy_max
from collections.abc import Iterable
from typing import Generator
from copy import deepcopy

from APPETITE.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree
from APPETITE.Diagnosers.SFLDT import SFLDT

# The diagnosers dictionary - format: {diagnoser name: (diagnoser class, (diagnoser default parameters tuple))}
diagnosers_dict = {
    "SFLDT": (SFLDT, ("faith", ))
}

class Fixer:
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series,
                 diagnoser_name: str = "SFLDT",
                 diagnoser_parameters: tuple[object] = None
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
        self.X = X
        self.y = y
        diagnoser_name, diagnoser_default_parameters = diagnosers_dict[diagnoser_name]
        if diagnoser_parameters is None:
            diagnoser_parameters = diagnoser_default_parameters
        if not isinstance(diagnoser_parameters, Iterable):
            diagnoser_parameters = (diagnoser_parameters, )
        self.diagnoser = diagnoser_name(self.mapped_tree, self.X, self.y, *diagnoser_parameters)
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
        
        Parameters:
            faulty_node_index (int): The index of the faulty node.
            data_reached_faulty_node (DataFrame): The data that reached the faulty node.
        """
        reached_labels = self.y[data_reached_faulty_node.index]
        most_common_class = reached_labels.value_counts().idxmax()

        # Make the most common class the class with the max count in the node
        values = self.mapped_tree.sklearn_tree_model.tree_.value[faulty_node_index]
        max_value_count = numpy_max(values)
        old_values = values[0]
        values[0][most_common_class] = max_value_count + 1
        print(f"Faulty node {faulty_node_index} (terminal) class changed from {old_values} to {values}")
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
        node_feature_average_differece = node_feature_average_after_drift - node_feature_average_before_drift
        new_threshold = faulty_node.threshold + node_feature_average_differece
        print(f"Faulty node {faulty_node_index} (Numeric) threshold changed from {faulty_node.threshold} to {new_threshold}")
        self.mapped_tree.sklearn_tree_model.tree_.threshold[faulty_node_index] = new_threshold

    def _fix_categorical_faulty_node(self,
                                    faulty_node_index: int,
                                    data_reached_faulty_node: DataFrame
     ) -> None:
          """
          Fix a categorical faulty node.
          The fix is done by flipping the switch of the condition in it.
    
          Parameters:
                faulty_node_index (int): The index of the faulty node.
                data_reached_faulty_node (DataFrame): The data that reached the faulty node.
          """
          raise NotImplementedError
          
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
            faulty_node_feature_type = "numeric" if is_numeric_dtype(data_reached_faulty_node[faulty_node.feature]) else "categorical"
        if faulty_node_feature_type == "numeric":
            self._fix_numeric_faulty_node(faulty_node_index, data_reached_faulty_node)
        else:
            self._fix_categorical_faulty_node(faulty_node_index, data_reached_faulty_node)
    
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
     ) -> MappedDecisionTree:
        """
        Fix the decision tree under the assumption that there is a single faulty node in the tree.

        Parameters:
            faulty_node (int): The index of the faulty node. If None, the faulty node will be detected using the diagnoser.
        Returns:
            MappedDecisionTree: The fixed decision tree.
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
        return self._create_fixed_mapped_tree()

    def fix_multiple_faults(self) -> MappedDecisionTree:
        """
        Fix all the faulty nodes in the decision tree.

        Returns:
            MappedDecisionTree: The fixed decision tree.
        """
        if self.tree_already_fixed:
            return self.mapped_tree
        raise NotImplementedError
        # TBC
        return self._create_fixed_mapped_tree()