from .ADiagnoser import *
from .STAT_SFLDT import STAT_SFLDT, SFLDT_DEFAULT_SIMILARITY_MEASURES
from .FuzzySFLDT import FuzzySFLDT

class STAT_FuzzySFLDT(STAT_SFLDT):
    """
    The diagnoser that combines the STAT and Fuzzy SFLDT diagnosers.
    """

    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series,
                 similarity_measure: str = SFLDT_DEFAULT_SIMILARITY_MEASURES
    ):
        """
        Initialize the diagnoser.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        similarity_measure (str): The similarity measure to use.
        """
        super().__init__(mapped_tree, X, y, similarity_measure)
        self.sfldt = FuzzySFLDT(mapped_tree, X, y, similarity_measure)