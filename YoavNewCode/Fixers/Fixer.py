import pandas as pd
import numpy as np
from collections.abc import Iterable

from YoavNewCode.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree
from YoavNewCode.Diagnosers.SFLDT import SFLDT

# The diagnosers dictionary - format: {diagnoser name: (diagnoser class, (diagnoser default parameters tuple))}
diagnosers_dict = {
    "SFLDT": (SFLDT, ("faith", ))
}

class Fixer:
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: pd.DataFrame,
                 y: pd.Series,
                 diagnoser_name: str = "SFLDT",
                 diagnoser_parameters: tuple[object] = None
    ):
        """
        Initialize the Fixer.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (pd.DataFrame): The data.
        y (pd.Series): The target column.
        diagnoser_name (str): The diagnoser name.
        diagnoser_parameters (tuple[object]): The diagnoser parameters.
        """
        self.mapped_tree = mapped_tree
        self.X = X
        self.y = y
        diagnoser_name, diagnoser_default_parameters = diagnosers_dict[diagnoser_name]
        if diagnoser_parameters is None:
            diagnoser_parameters = diagnoser_default_parameters
        if not isinstance(diagnoser_parameters, Iterable):
            diagnoser_parameters = (diagnoser_parameters, )
        self.diagnoser = diagnoser_name(self.mapped_tree, self.X, self.y, *diagnoser_parameters)

    def fix_single_fault(self) -> MappedDecisionTree:
        """
        Fix the decision tree under the assumption that there is a single faulty node in the tree.

        Returns:
            MappedDecisionTree: The fixed decision tree.
        """
        # Get the nodes according to their "faulty value"
        faulty_nodes = self.diagnoser.get_diagnosis()
        faulty_node = faulty_nodes[0]

        # Fix the faulty node
        # TODO: Implement the fix