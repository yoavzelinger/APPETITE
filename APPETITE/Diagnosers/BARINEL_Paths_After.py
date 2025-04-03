from pandas import Series

from .BARINEL_Paths import BARINEL_Paths, get_fuzzy_error

class BARINEL_Paths_After(BARINEL_Paths):
    def get_fuzzy_error_data(self,
                       before_accuracy_vector: Series,
                       current_accuracy_vector: Series
    ) -> tuple[Series, float, float]:
        """
        Get the fuzzy data.
        The fuzzy error is calculated as the current error the drift.
        
        Parameters:
        before_accuracy_vector (Series): The accuracy before the drift.
        current_accuracy_vector (Series): The accuracy after the drift.
        
        Returns:
        Series: The fuzzy error.
        float: The error average.
        float: The error standard deviation.
        """
        before_error_vector, current_error_vector = get_fuzzy_error(before_accuracy_vector), get_fuzzy_error(current_accuracy_vector)
        error_average, error_std = before_error_vector.mean(), before_error_vector.std()
        return current_error_vector, error_average, error_std