from abc import ABC, abstractmethod
from pandas import DataFrame, Series

from APPETITE.Constants import SINGLE_DIAGNOSER_TYPE_NAME, MULTIPLE_DIAGNOSER_TYPE_NAME, DIAGNOSER_TYPES
from APPETITE.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree

class ADiagnoser(ABC):
    diagnoser_type = None # Can be "single" or "multiple"
    def get_diagnoser_type(self) -> str:
        """
        Get the diagnoser type.
        
        Returns:
        str: The diagnoser type.
        """
        assert self.diagnoser_type in DIAGNOSER_TYPES, "DIAGNOSER_TYPE must be defined"
        return self.diagnoser_type
    
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series
    ):
        """
        Initialize the STAT diagnoser.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        """
        self.mapped_tree = mapped_tree
        self.X_after = X
        self.y_after = y
        self.diagnoses = None

    def get_diagnoses_without_ranks(self,
                                    diagnoses: list[int] | list[tuple[int, float]] | list[list[int]] | list[tuple[list[int], float]]
    ):
        return [diagnosis for diagnosis, _ in diagnoses]
    
    def sort_diagnoses(self):
        self.diagnoses.sort(key=lambda diagnosis: diagnosis[1], reverse=True)
        
    @abstractmethod
    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
                      diagnoses: list[int] | list[tuple[int, float]] | list[list[int]] | list[tuple[list[int], float]] = None
     ) -> list[int] | list[tuple[int, float]] | list[list[int]] | list[tuple[list[int], float]]:
        """
        Get the diagnosis of the nodes.
        Each diagnosis consist nodes.
        The diagnoses ordered by their rank.

        Parameters:
        retrieve_ranks (bool): Whether to return the diagnosis ranks.
        diagnoses (list[int] | list[tuple[int, float]] | list[list[int]] | list[tuple[list[int], float]]): Optional - Given diagnoses

        Returns:
        list[int] | list[tuple[int, float]] | list[list[int]] | list[tuple[list[int], float]]: The diagnosis (can be single or multiple). If retrieve_ranks is True, the diagnosis will be a list of tuples,
          where the first element is the diagnosis and the second is the rank.
        """
        if diagnoses is None:
            diagnoses = self.diagnoses
        return diagnoses if retrieve_ranks else self.get_diagnoses_without_ranks(diagnoses)