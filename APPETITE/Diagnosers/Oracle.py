import numpy as np
import pandas as pd
from collections import defaultdict

from .ADiagnoser import *

class Oracle(ADiagnoser):
    def __init__(self, 
                 mapped_model: ATreeBasedMappedModel,
                 X: pd.DataFrame,
                 y: pd.Series,
                 actual_faulty_features: list[str]
    ):
        """
        Initialize the Oracle diagnoser.
        
        Parameters:
        mapped_model (ATreeBasedMappedModel): The mapped model.
        X (DataFrame): The data.
        y (Series): The target column.
        actual_faulty_features (list[str]): The actual faulty features.
        """
        super().__init__(mapped_model, X, y)

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
            node: TreeNodeComponent
            for node in self.mapped_model:
                if node.feature in self.actual_faulty_features:
                    actual_faulty_nodes.append(node.get_index())
            self.diagnoses = [(actual_faulty_nodes, 1)]
        return super().get_diagnoses(retrieve_ranks)