from pandas import DataFrame, Series


from .BARINEL_Features import BARINEL_Features

class BARINEL_Features_Paths_After(BARINEL_Features):
    def fill_spectra_and_error_vector(self, 
                                      X: DataFrame, 
                                      y: Series
     ) -> None:
        return super().fill_spectra_and_error_vector(X, y, use_fuzzy_error=False)
    
    def get_fuzzy_error_data(self,
                       before_accuracy_vector: Series,
                       current_accuracy_vector: Series
    ) -> tuple[Series, float, float]:
        return super().get_fuzzy_error_data(before_accuracy_vector, current_accuracy_vector, diagnosis_algorithm="SFLDT", use_fuzzy_error=False)
    
    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
                      retrieve_spectra_indices: bool = False
     ) -> list[list[int]] | list[tuple[list[int], float]]:
        return super().get_diagnoses(retrieve_ranks, retrieve_spectra_indices, diagnosis_algorithm="SFLDT")