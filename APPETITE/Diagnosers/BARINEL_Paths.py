from numpy import zeros
from pandas import DataFrame, Series

from APPETITE.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree

from .BARINEL import *

from APPETITE.Constants import BARINEL_PATHS_ERROR_STD_THRESHOLD

def get_fuzzy_error(accuracy: float) -> float:
    """
    Get the fuzzy error based on the accuracy.
    
    Parameters:
    accuracy (float): The accuracy.
    
    Returns:
    float: The fuzzy error.
    """
    return 1 - accuracy

class BARINEL_Paths(BARINEL):
    
    diagnoser_type = MULTIPLE_DIAGNOSER_TYPE_NAME
    
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series,
    ):
        """
        Initialize the SFLDT diagnoser.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        similarity_measure (str): The similarity measure to use.
        """
        super().__init__(mapped_tree, X, y)

    @abstractmethod
    def get_fuzzy_data(self,
                       before_accuracy_vector: Series,
                       current_accuracy_vector: Series
    ) -> tuple[Series, float, float]:
        """
        Get the fuzzy data.
        
        Parameters:
        before_accuracy_vector (Series): The accuracy before the drift.
        current_accuracy_vector (Series): The accuracy after the drift.
        
        Returns:
        Series: The fuzzy error.
        float: The error average.
        float: The error threshold.
        """
        raise NotImplementedError("The method 'get_fuzzy_data' must be implemented")

    def fill_spectra_and_error_vector(self, 
                                      X: DataFrame, 
                                      y: Series
     ) -> None:
        """
        Fill the spectra and error vector with paths.
        """
        paths_dict = {}
        node_indicator = self.mapped_tree.sklearn_tree_model.tree_.decision_path(X.to_numpy(dtype="float32"))
        for sample_id in range(self.sample_count):
            participated_nodes = node_indicator.indices[
                node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
            ]
            spectra_participation = []
            path_terminal = None
            for node in map(self.mapped_tree.get_node, participated_nodes):
                node_spectra_index = node.spectra_index
                spectra_participation.append(node_spectra_index)
                if node.is_terminal():
                    path_terminal = node
            if path_terminal is None:
                raise ValueError(f"No terminal node found for sample {sample_id}. Paths: {participated_nodes}")
            path_before_accuracy = path_terminal.correct_classifications_count / path_terminal.reached_samples_count
            spectra_participation = tuple(spectra_participation)
            classified_correctly_count, total_count, _ = paths_dict.get(spectra_participation, (0, 0, 0))
            paths_dict[spectra_participation] = (classified_correctly_count + int(path_terminal.class_name == y[sample_id]), total_count + 1, path_before_accuracy)
        
        self.paths_count = len(paths_dict)
        self.spectra = zeros((self.node_count, self.paths_count))
        self.error_vector = zeros(self.paths_count)
        before_accuracy_vector, current_accuracy_vector = zeros(self.paths_count), zeros(self.paths_count)
        for path_index, (path, (classified_correctly_count, total_count, path_before_accuracy)) in enumerate(paths_dict.items()):
            for node_spectra_index in path:
                self.spectra[node_spectra_index, path_index] = 1
                node = self.mapped_tree.get_node(node_spectra_index, use_spectra_index=True)
            path_current_accuracy = classified_correctly_count / total_count
            before_accuracy_vector[path_index] = path_before_accuracy
            current_accuracy_vector[path_index] = path_current_accuracy
            assert path_before_accuracy >= 0 and path_before_accuracy <= 1, f"Path before accuracy is {path_before_accuracy}"
            assert path_current_accuracy >= 0 and path_current_accuracy <= 1, f"Path current accuracy is {path_current_accuracy} ({classified_correctly_count} / {total_count})"
        fuzzy_error_vector, error_average, error_std = self.get_fuzzy_data(before_accuracy_vector, current_accuracy_vector)
        error_threshold = error_average + BARINEL_PATHS_ERROR_STD_THRESHOLD * error_std
        self.error_vector = (fuzzy_error_vector >= error_threshold).astype(int)
        assert self.error_vector.sum() > 0, f"No path with error above the threshold {error_threshold} (average: {error_average}). The largest error is {max(fuzzy_error_vector)}"