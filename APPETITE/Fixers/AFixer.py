from abc import ABC, abstractmethod
import pandas as pd
from copy import deepcopy

from sklearn.tree import DecisionTreeClassifier

from APPETITE.ModelMapping.ATreeBasedMappedModel import ATreeBasedMappedModel
from APPETITE.ModelMapping.TreeNodeComponent import TreeNodeComponent

from APPETITE.Diagnosers import *

class AFixer(ABC):
    alias = None
    def __init__(self, 
                 mapped_model: ATreeBasedMappedModel,
                 X: pd.DataFrame,
                 y: pd.Series,
                 faulty_nodes_indices: list[int],
                 X_prior: pd.DataFrame = None,
                 y_prior: pd.Series = None,
                 sklearn_model: DecisionTreeClassifier = None
    ):
        """
        Initialize the Fixer.
        
        Parameters:
        mapped_model (ATreeBasedMappedModel): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        faulty_nodes_indices (list[int]): The indices of the faulty nodes.
        sklearn_model (DecisionTreeClassifier, optional): The sklearn decision tree model. Defaults to None (Taken from mapped_model).
        """
        assert self.alias is not None, "Alias must be set to a fixer class"

        self.original_mapped_model = deepcopy(mapped_model)
        self.feature_types = mapped_model.data_feature_types
        self.X = X
        self.y = y
        self.faulty_nodes_indices = faulty_nodes_indices
        self.fixed_tree: DecisionTreeClassifier = None

        self.X_prior = X_prior
        self.y_prior = y_prior
        
        self.sklearn_model = sklearn_model if sklearn_model else mapped_model.model
        
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
        faulty_node = self.original_mapped_model[faulty_node_index]
        return faulty_node.get_data_reached_node(self.X, self.y, allow_empty=False)
    
    @abstractmethod
    def fix_tree(self) -> DecisionTreeClassifier:
        """
        Fix the decision tree.

        Returns:
            DecisionTreeClassifier: The fixed decision tree.
        """
        assert self.fixed_tree, "The tree wasn't fixed yet"

        return self.fixed_tree