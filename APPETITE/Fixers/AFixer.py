from abc import ABC, abstractmethod
import pandas as pd
from copy import deepcopy

from sklearn.tree import DecisionTreeClassifier

from APPETITE.MappedDecisionTree import MappedDecisionTree

from APPETITE.Diagnosers import *

class AFixer(ABC):
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: pd.DataFrame,
                 y: pd.Series,
                 diagnoser__class_name: str,
                 diagnoser_parameters: dict[str, object],
                 diagnoser_output_name: str = None
    ):
        """
        Initialize the Fixer.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        diagnoser_name (str): The diagnoser name.
        diagnoser_parameters ( dict[str, object]): The diagnoser parameters.
        diagnoser_output_name (str): The diagnoser output name.
        """
        self.original_mapped_tree = mapped_tree
        self.feature_types = mapped_tree.data_feature_types
        self.X = X
        self.y = y
        diagnoser_class = get_diagnoser(diagnoser__class_name)
        self.diagnoser_output_name = diagnoser_output_name if diagnoser_output_name else diagnoser__class_name
        self.diagnoser: ADiagnoser = diagnoser_class(self.original_mapped_tree, self.X, self.y, **diagnoser_parameters)
        self.diagnoses = self.diagnoser.get_diagnoses()
        self.faulty_nodes: list[int] = self.diagnoses[0]
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
    def fix_tree(self) -> tuple[DecisionTreeClassifier, list[int]]:
        """
        Fix the decision tree.

        Returns:
            DecisionTreeClassifier: The fixed decision tree.
            list[int]: The indices of the faulty nodes.
        """
        assert self.fixed_tree, "The tree wasn't fixed yet"

        return self.fixed_tree, self.faulty_nodes