from numpy import zeros, clip
from pandas import DataFrame, Series

from APPETITE.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree

from .BARINEL import *

from APPETITE.Constants import BARINEL_PATHS_ACCURACY_THRESHOLD

def get_fuzzy_error(accuracy: float) -> float:
    """
    Get the fuzzy error based on the accuracy.
    
    Parameters:
    accuracy (float): The accuracy.
    
    Returns:
    float: The fuzzy error.
    """
    return clip(1 - accuracy, 0, 1)

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
        for path_index, (path, (classified_correctly_count, total_count, path_before_accuracy)) in enumerate(paths_dict.items()):
            path_before_accuracy = -1
            for node_spectra_index in path:
                self.spectra[node_spectra_index, path_index] = 1
                node = self.mapped_tree.get_node(node_spectra_index, use_spectra_index=True)
            path_current_accuracy = classified_correctly_count / total_count
            # accuracy_difference = path_current_accuracy - path_before_accuracy
            # error = get_fuzzy_error(accuracy_difference)
            error = get_fuzzy_error(path_current_accuracy)
            self.error_vector[path_index] = error
        accuracy_threshold = min(BARINEL_PATHS_ACCURACY_THRESHOLD, max(self.error_vector))
        self.error_vector = (self.error_vector >= accuracy_threshold).astype(int)