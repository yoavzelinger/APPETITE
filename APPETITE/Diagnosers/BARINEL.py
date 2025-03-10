from numpy import ndarray, array as np_array
from pandas import DataFrame, Series

from APPETITE.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree
from .barinel_utils import *
from .ADiagnoser import *
from .SFLDT import SFLDT

from APPETITE.Constants import GRADIENT_STEP

def get_barinel_diagnoses(spectra: ndarray,
                          error_vector: ndarray
 ) -> list[tuple[list[int], float]]:
    """
    Perform the Barinel diagnosis algorithm on the given spectrum.

    Parameters:
    spectrum (list[list[int]]): The spectrum.

    Returns:
    list[tuple[list[int], float]]: The diagnoses with their corresponding ranks.
    """
    spectrum = list(map(lambda spectra_vector_pair: spectra_vector_pair[0] + [spectra_vector_pair[1]], zip(spectra.T.tolist(), error_vector.tolist())))
    diagnoses, _ = _barinel_diagnosis(spectrum, [])
    # diagnoses = _rank_diagnoses(spectrum, diagnoses, GRADIENT_STEP)
    diagnoses = list(map(np_array, diagnoses))
    diagnoses = rank_diagnoses(diagnoses, np_array(spectrum), error_vector)
    diagnoses = [(diagnosis[0], diagnosis[1]) for diagnosis in diagnoses]
    return diagnoses

class BARINEL(SFLDT):

    diagnoser_type = MULTIPLE_DIAGNOSER_TYPE_NAME

    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series
    ):
        super().__init__(mapped_tree, X, y)

    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
                      retrieve_spectra_indices: bool = False
     ) -> list[list[int]] | list[tuple[list[int], float]]:
        """
        Get the diagnosis of the nodes.
        The diagnosis consists the nodes ordered.

        Parameters:
        retrieve_ranks (bool): Whether to return the diagnosis ranks.
        retrieve_spectra_indices (bool): Whether to return the spectra indices or the node indices.

        Returns:
        list[int] | list[tuple[int, float]]: The diagnosis. If retrieve_ranks is True, the diagnosis will be a list of tuples,
          where the first element contains the indices of the faulty nodes and the second is the similarity rank.
        """
        if self.diagnoses is None:
            self.diagnoses = get_barinel_diagnoses(self.spectra, self.error_vector)
            self.sort_diagnoses()
        return super().get_diagnoses(retrieve_ranks, retrieve_spectra_indices)
        
