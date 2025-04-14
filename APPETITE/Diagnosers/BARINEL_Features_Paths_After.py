from pandas import DataFrame, Series


from .BARINEL_Features import BARINEL_Features

class BARINEL_Features_Paths_After(BARINEL_Features):
    def fill_spectra_and_error_vector(self, 
                                      X: DataFrame, 
                                      y: Series
     ) -> None:
        return super().fill_spectra_and_error_vector(X, y, use_fuzzy_error=True)
    
    def get_fuzzy_error_data(self,
                       before_accuracy_vector: Series,
                       current_accuracy_vector: Series
    ) -> tuple[Series, float, float]:
        return super().get_fuzzy_error_data(before_accuracy_vector, current_accuracy_vector, barinel_paths_type="AFTER")