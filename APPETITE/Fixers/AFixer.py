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
        self.diagnoser: ADiagnoser = diagnoser_class(self.mapped_tree, self.X, self.y, **diagnoser_parameters)
        self.faulty_nodes = None    # List of sk_indices of the faulty nodes; Lazy evaluation
        self.tree_already_fixed = False
        self.diagnoser_output_name = diagnoser_output_name if diagnoser_output_name else diagnoser__class_name

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
        assert self.faulty_nodes is not None, "The faulty nodes weren't identified yet"
        assert self.tree_already_fixed, "The tree wasn't fixed yet"

        return self._create_fixed_mapped_tree(), self.faulty_nodes