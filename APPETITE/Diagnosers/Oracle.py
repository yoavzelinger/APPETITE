import numpy as np
import pandas as pd
from collections import defaultdict

from APPETITE import Constants as constants
from APPETITE.MappedDecisionTree import MappedDecisionTree

from .ADiagnoser import ADiagnoser

class Oracle(ADiagnoser):
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: pd.DataFrame,
                 y: pd.Series,
                 actual_faulty_features: list[str]
    ):
        """
        Initialize the Oracle diagnoser.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        actual_faulty_features (list[str]): The actual faulty features.
        """
        super().__init__(mapped_tree, X, y)

        self.actual_faulty_features = actual_faulty_features

    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
     ) -> list[list[int]] | list[tuple[list[int], float]]:
        """
        Get the diagnosis of the nodes.
        Each diagnosis consist nodes.
        The diagnoses ordered by their rank.

        Parameters:
        retrieve_ranks (bool): Whether to return the diagnosis ranks.

        Returns:
        list[list[int]] | list[tuple[list[int], float]]: The diagnosis (can be single or multiple). If retrieve_ranks is True, the diagnosis will be a list of tuples,
          where the first element is the diagnosis and the second is the rank.
        """
        if self.diagnoses is None:    
            actual_faulty_nodes = []
            for node_index, node in self.mapped_tree.tree_dict.items():
                if node.feature in self.actual_faulty_features:
                    actual_faulty_nodes.append(node_index)
            self.diagnoses = [(actual_faulty_nodes, 1)]

        self.sort_diagnoses()
        return self.diagnoses if retrieve_ranks else self.get_diagnoses_without_ranks(self.diagnoses)