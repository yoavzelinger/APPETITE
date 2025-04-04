from numpy import ndarray, array as np_array, concatenate as np_concatenate
from pandas import DataFrame, Series

from APPETITE.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree
from .barinel_utils import *
from .ADiagnoser import *
from .SFLDT import SFLDT
from .FuzzySFLDT import FuzzySFLDT

def get_barinel_diagnoses(spectra: ndarray,
                          error_vector: ndarray,
                          components_prior_probabilities: ndarray = None,
                          error_threshold: float = None
 ) -> list[tuple[list[int], float]]:
    """
    Perform the Barinel diagnosis algorithm on the given spectrum.

    Parameters:
    spectra (ndarray): The spectrum.
    error_vector (ndarray): The error vector.
    components_prior_probabilities (ndarray): The components prior probabilities.
    threshold (float): The threshold for the error vector.

    Returns:
    list[tuple[list[int], float]]: The diagnoses with their corresponding ranks.
    """
    discrete_error_vector = error_vector if error_threshold is None else (error_vector >= error_threshold).astype(int)
    assert all([error in [0, 1] for error in discrete_error_vector]), f"The error vector must be binary (for candidation). Provided threshold: {error_threshold}"
    assert sum(discrete_error_vector) > 0, f"No path with error above the threshold {error_threshold} (average: {error_vector.mean()}). The largest error is {max(error_vector)}"
    spectrum = list(map(lambda spectra_vector_pair: spectra_vector_pair[0] + [spectra_vector_pair[1]], zip(spectra.T.tolist(), discrete_error_vector.tolist())))
    diagnoses, _ = get_candidates(spectrum)
    diagnoses = list(map(np_array, diagnoses))
    assert len(diagnoses) > 0, "No candidate diagnoses found"
    spectrum = np_concatenate((np_array(spectrum)[:, :-1], np_array([error_vector]).T), axis=1)
    diagnoses = rank_diagnoses(spectrum, diagnoses, components_prior_probabilities)
    diagnoses = [(diagnosis[0], diagnosis[1]) for diagnosis in diagnoses]
    return diagnoses

class BARINEL(SFLDT):

    diagnoser_type = MULTIPLE_DIAGNOSER_TYPE_NAME

    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series
    ):
        self.components_prior_probabilities = None
        self.threshold = None
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
        components_prior_probabilities (ndarray): The components prior probabilities.
        threshold (float): The threshold for the error vector

        Returns:
        list[int] | list[tuple[int, float]]: The diagnosis. If retrieve_ranks is True, the diagnosis will be a list of tuples,
          where the first element contains the indices of the faulty nodes and the second is the similarity rank.
        """
        if self.diagnoses is None:
            self.update_fuzzy_participation()
            self.diagnoses = get_barinel_diagnoses(spectra=self.spectra, error_vector=self.error_vector, components_prior_probabilities=self.components_prior_probabilities, error_threshold=self.threshold)
            self.sort_diagnoses()
        return super().get_diagnoses(retrieve_ranks, retrieve_spectra_indices)
        
