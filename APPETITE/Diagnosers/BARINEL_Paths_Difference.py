from pandas import Series

from .BARINEL_Paths import BARINEL_Paths, get_fuzzy_error

class BARINEL_Paths_Difference(BARINEL_Paths):
    def get_fuzzy_data(self,
                       before_accuracy_vector: Series,
                       current_accuracy_vector: Series
    ) -> tuple[Series, float, float]:
        """
        Get the fuzzy data.
        The fuzzy error is calculated as the difference between the accuracy before and after the drift.
        
        Parameters:
        before_accuracy_vector (Series): The accuracy before the drift.
        current_accuracy_vector (Series): The accuracy after the drift.
        
        Returns:
        Series: The fuzzy error.
        float: The error average.
        float: The error standard deviation.
        """
        error_vector = get_fuzzy_error(current_accuracy_vector - before_accuracy_vector)
        error_average, error_std = error_vector.mean(), error_vector.std()
        return error_vector, error_average, error_std