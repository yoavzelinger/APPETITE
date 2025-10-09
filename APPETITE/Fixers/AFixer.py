from abc import ABC, abstractmethod
import pandas as pd
from copy import deepcopy

from APPETITE.MappedDecisionTree import MappedDecisionTree

from APPETITE.Diagnosers import *

class AFixer(ABC):
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: pd.DataFrame,
                 y: pd.Series,
                 diagnoser__class_name: str,
                 diagnoser_parameters: dict[str, object],
                 diagnoser_output_name: str=None
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
        self.mapped_tree = deepcopy(mapped_tree)
        self.feature_types = mapped_tree.data_feature_types
        self.X = X
        self.y = y
        diagnoser_class = get_diagnoser(diagnoser__class_name)
        self.diagnoser_output_name = diagnoser_output_name if diagnoser_output_name else diagnoser__class_name
        self.diagnoser: ADiagnoser = diagnoser_class(self.mapped_tree, self.X, self.y, **diagnoser_parameters)
        self.diagnoses = self.diagnoser.get_diagnoses()
        self.faulty_nodes: list[int] = self.diagnoses[0]
        self.tree_already_fixed = False

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
        faulty_node = self.mapped_tree.get_node(faulty_node_index)
        filtered_data = faulty_node.get_data_reached_node(self.X)
        while filtered_data.empty and faulty_node.parent is not None:
            # Get the data that reached the parent node
            faulty_node = faulty_node.parent
            filtered_data = faulty_node.get_data_reached_node(self.X)
        return filtered_data

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
        return fixed_mapped_decision_tree
    
    @abstractmethod
    def fix_tree(self
     ) -> tuple[MappedDecisionTree, list[int]]:
        """
        Fix the decision tree.

        Returns:
            MappedDecisionTree: The fixed decision tree.
            list[int]: The indices of the faulty nodes.
        """
        assert self.tree_already_fixed, "The tree wasn't fixed yet"

        return self._create_fixed_mapped_tree(), self.faulty_nodes