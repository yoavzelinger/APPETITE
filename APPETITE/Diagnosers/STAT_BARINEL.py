from numpy import array as np_array
from .ADiagnoser import *
from .STAT import STAT
from .BARINEL import BARINEL

from APPETITE import Constants as constants

class STAT_BARINEL(ADiagnoser):
    """
    The diagnoser that combines the STAT and BARINEL diagnosers.
    """

    diagnoser_type = constants.MULTIPLE_DIAGNOSER_TYPE_NAME

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
        self.stat = STAT(mapped_tree, X, y)
        self.barinel = BARINEL(mapped_tree, X, y)

    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
                      retrieve_spectra_indices: bool = False
     ) -> list[int] | list[tuple[int, float]]:
        """
        Get the diagnoses.
        For the prior probability to the BARINEL diagnoses, we use the average STAT of all the nodes in each diagnosis.
        Thus, the diagnoses is calculated as the multiplication of the diagnoses of the STAT and BARINEL diagnosers.

        
        Returns:
        list[int] | list[tuple[int, float]]: The diagnoses. If retrieve_ranks is True, the diagnoses will be a list of tuples with the node index and the rank.
        """
        if self.diagnoses is None:
            stat_diagnoses = self.stat.get_diagnoses(retrieve_ranks=True)
            stat_diagnoses.sort()
            self.barinel.components_prior_probabilities = np_array([diagnosis[1] for diagnosis in stat_diagnoses])
            self.diagnoses = self.barinel.get_diagnoses(retrieve_ranks=True, retrieve_spectra_indices=retrieve_spectra_indices)
            self.sort_diagnoses()
        return super().get_diagnoses(retrieve_ranks)