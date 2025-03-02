from numpy import zeros
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
    return 1 - accuracy

class BARINEL_Paths(BARINEL):
    
    diagnoser_type = MULTIPLE_DIAGNOSER_NAME
    
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

    def fill_spectra_and_error_vector_with_paths(self, 
                                      X: DataFrame, 
                                      y: Series
     ) -> None:
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
            spectra_participation = tuple(spectra_participation)
            classified_correctly_count, total_count = paths_dict.get(spectra_participation, (0, 0))
            paths_dict[spectra_participation] = (classified_correctly_count + int(path_terminal.class_name == y[sample_id]), total_count + 1)
        
        self.paths_count = len(paths_dict)
        self.spectra = zeros((self.node_count, self.paths_count))
        self.error_vector = zeros(self.paths_count)
        for path_index, (path, (classified_correctly_count, total_count)) in enumerate(paths_dict.items()):
            for node_spectra_index in path:
                self.spectra[node_spectra_index, path_index] = 1
            error = get_fuzzy_error(classified_correctly_count / total_count)
            self.error_vector[path_index] = int(error > BARINEL_PATHS_ACCURACY_THRESHOLD)