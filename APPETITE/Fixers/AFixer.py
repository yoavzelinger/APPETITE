from abc import ABC, abstractmethod
import pandas as pd
from copy import deepcopy

from sklearn.tree import DecisionTreeClassifier

from APPETITE.MappedDecisionTree import MappedDecisionTree

from APPETITE.Diagnosers import *

class AFixer(ABC):
    alias = None
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: pd.DataFrame,
                 y: pd.Series,
                 faulty_nodes: list[int]
    ):
        """
        Initialize the Fixer.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        faulty_nodes (list[int]): The indices of the faulty nodes.
        """
        assert self.alias is not None, "Alias must be set to a fixer class"

        self.original_mapped_tree = deepcopy(mapped_tree)
        self.feature_types = mapped_tree.data_feature_types
        self.X = X
        self.y = y
        self.faulty_nodes = faulty_nodes
        self.fixed_tree: DecisionTreeClassifier = None

    def _filter_data_reached_fault(self,
                                  faulty_node_index: int
        ) -> pd.DataFrame:
        """
        Filter the data that reached the faulty nodes.

        Parameters:
            faulty_nodes_count (int): The number of faulty nodes.

        Returns:
            DataFrame: The data that reached the faulty nodes.
        """
        faulty_node = self.original_mapped_tree.get_node(faulty_node_index)
        filtered_data = faulty_node.get_data_reached_node(self.X, self.y)
        while filtered_data[0].empty and faulty_node.parent is not None:
            # Get the data that reached the parent node
            faulty_node = faulty_node.parent
            filtered_data = faulty_node.get_data_reached_node(self.X, self.y)
        return filtered_data
    
    @abstractmethod
    def fix_tree(self) -> DecisionTreeClassifier:
        """
        Fix the decision tree.

        Returns:
            DecisionTreeClassifier: The fixed decision tree.
        """
        assert self.fixed_tree, "The tree wasn't fixed yet"

        return self.fixed_tree