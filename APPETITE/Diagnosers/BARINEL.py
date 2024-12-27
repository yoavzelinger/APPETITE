from pandas import DataFrame, Series

# from barinel_utils.methods_for_diagnosis import diagnosis_0 as _barinel_diagnosis
from .barinel_utils import *
from .SFLDT import SFLDT

def get_barinel_diagnoses(spectra: DataFrame,
                          error_vector: Series) -> list[list[int]]:
    """
    Perform the Barinel diagnosis algorithm on the given spectrum.

    Parameters:
    spectrum (list[list[int]]): The spectrum.

    Returns:
    list[list[int]]: All the diagnoses.
    """
    extended_spectra = list(map(lambda spectra_vector_pair: spectra_vector_pair[0] + [spectra_vector_pair[1]], zip(spectra.values.tolist(), error_vector.tolist())))
    diagnoses, _ = _barinel_diagnosis(extended_spectra, [])
    return diagnoses

class BARINEL(SFLDT):
    def __init__(mapped_tree, X, y):
        super().__init__(mapped_tree, X, y)

    def get_diagnosis(self,
                      retrieve_spectra_indices: bool = False
    ) -> list[list[int]]:
        """
        Get the diagnosis of the nodes.
        The diagnosis consists the nodes ordered.

        Parameters:
        retrieve_spectra_indices (bool): Whether to return the spectra indices or the node indices.

        Returns:
        list[list[int]]: The diagnoses.
        """
        diagnoses = get_barinel_diagnoses(self.spectra, self.error_vector)
        if retrieve_spectra_indices: 
            return diagnoses
        convert_index_func = self.mapped_tree.convert_spectra_index_to_node_index
        for diagnosis in diagnoses:
            for diagnosis_node_index, spectra_index in enumerate(diagnosis):
                diagnosis[diagnosis_node_index] = convert_index_func(spectra_index)
        return diagnoses