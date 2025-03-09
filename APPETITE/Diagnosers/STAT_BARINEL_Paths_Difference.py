from .STAT_BARINEL import *

from .BARINEL_Paths_Difference import BARINEL_Paths_Difference

class STAT_BARINEL_Paths_Difference(STAT_BARINEL):
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series,
                 stat_type: str = BARINEL_STAT_TYPE
    ):
        """
        Initialize the diagnoser.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        """
        super().__init__(mapped_tree, X, y, stat_type)
        self.barinel = BARINEL_Paths_Difference(mapped_tree, X, y)