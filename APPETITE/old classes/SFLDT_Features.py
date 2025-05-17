from pandas import DataFrame, Series

from .BARINEL_Features import BARINEL_Features

class SFLDT_Features(BARINEL_Features):
    def fill_spectra_and_error_vector(self, 
                                      X: DataFrame, 
                                      y: Series
     ) -> None:
        return super().fill_spectra_and_error_vector(X, y, use_fuzzy_error=False)
    
    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
                      retrieve_spectra_indices: bool = False
     ) -> list[list[int]] | list[tuple[list[int], float]]:
        return super().get_diagnoses(retrieve_ranks, retrieve_spectra_indices, diagnosis_algorithm="SFLDT")