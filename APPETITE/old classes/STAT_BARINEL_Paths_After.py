from .STAT_BARINEL import *

from .BARINEL_Paths_After import BARINEL_Paths_After

class STAT_BARINEL_Paths_After(STAT_BARINEL):
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series
    ):
        """
        Initialize the diagnoser.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        """
        super().__init__(mapped_tree, X, y)
        self.barinel = BARINEL_Paths_After(mapped_tree, X, y)