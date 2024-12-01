import pandas as pd
import numpy as np

from YoavNewCode.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree

class SFLDT:
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: pd.DataFrame,
                 y: pd.Series,
                 similarity_measure: str = "faith"
    ):
        """
        Initialize the SFLDT diagnoser.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (pd.DataFrame): The data.
        y (pd.Series): The target column.
        similarity_measure (str): The similarity measure to use.
        """
        self.mapped_tree = mapped_tree
        self.node_count = mapped_tree.node_count
        self.sample_count = len(X)
        self.spectra = np.zeros((self.node_count, self.sample_count))
        self.error_vector = np.zeros(self.sample_count)
        self.similarity_measure = similarity_measure
        self.fill_spectra_and_error_vector(X, y)

    def get_faith_similarity(self,
                             index: int,
                             spectra_index: bool = True
     ) -> float:
        """
        Get the faith similarity of the node to the error vector.

        Parameters:
        index (int): The index of the node.
        spectra_index (bool): Whether the index is the index of the node in the mapped tree or the index of the node in the spectra matrix.

        Returns:
        float: The faith similarity of the node to the error vector.
        """
        n11, n10, n01, n00 = (
            self.get_errror_participation_count(index, spectra_index),
            self.get_accurate_participation_count(index, spectra_index),
            self.get_error_nonparticipation_count(index, spectra_index),
            self.get_accurate_nonparticipation_count(index, spectra_index)
        )
        return (n11 +  0.5 * n00) / (n11 + n10 + n01 + n00)
    
    similarity_measure_functions_dict = {
        "faith": get_faith_similarity
    }

    def fill_spectra_and_error_vector(self, 
                                      X: pd.DataFrame, 
                                      y: pd.Series
     ) -> None:
        """
        Fill the spectra matrix and the error vector.

        Parameters:
        X (pd.DataFrame): The data.
        y (pd.Series): The target column.
        """
        # Source: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#decision-path
        node_indicator = self.mapped_tree.decision_path(X)
        for sample_id in range(self.sample_count):
            participated_nodes = node_indicator.indices[
                node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
            ]
            for node in map(self.mapped_tree.get_node, participated_nodes):
                node_spectra_index = node.spectra_index
                self.spectra[node_spectra_index, sample_id] = 1
                if node.is_terminal():
                    error = node.class_name != y[sample_id]
                    self.error_vector[sample_id] = int(error)

    def get_diagnosis(self,
                      retrieve_spectra_indices: bool = False
     ) -> list[int]:
        """
        Get the diagnosis of the nodes.
        The diagnosis consists the nodes ordered by their similarity to the error vector (DESC).

        Parameters:
        similarity_measure (str): The similarity measure to use.
        retrieve_spectra_indices (bool): Whether to return the spectra indices or the node indices.

        Returns:
        list[int]: The diagnosis of the nodes. For each node, the spectra/node index, ordered by their similarity to the error vector (DESC).
        """
        similarity_measure_function = self.similarity_measure_functions_dict[self.similarity_measure]
        sotred_spectra_indices = sorted(range(self.node_count), key=lambda spectra_index: similarity_measure_function(spectra_index), reverse=True)
        if retrieve_spectra_indices:
            return sotred_spectra_indices
        return list(map(self.mapped_tree.get_node, sotred_spectra_indices))
    
    def get_errror_participation_count(self, 
                                       index: int, 
                                       spectra_index: bool = True
     ) -> int: # n_1,1
        """
        Get n_1,1 - the number of samples that the node participated in and were misclassified.

        Parameters:
        index (int): The index of the node.
        spectra_index (bool): Whether the index is the index of the node in the mapped tree or the index of the node in the spectra matrix.

        Returns:
        int: The number of samples that the node participated in and were misclassified.
        """
        if not spectra_index:
            index = self.mapped_tree.get_node(index).spectra_index
        
        return self.error_vector @ self.spectra[index]
    
    def get_accurate_participation_count(self, 
                                         index: int, 
                                         spectra_index: bool = True
     ) -> int: # n_1,0
        """
        Get n_1,0 - the number of samples that the node participated in and were classified correctly.
        
        Parameters:
        index (int): The index of the node.
        spectra_index (bool): Whether the index is the index of the node in the mapped tree or the index of the node in the spectra matrix.
        
        Returns:
        int: The number of samples that the node participated in and were classified correctly.
        """
        if not spectra_index:
            index = self.mapped_tree.get_node(index).spectra_index
        
        return (1 - self.error_vector) @ self.spectra[index]
    
    def get_error_nonparticipation_count(self, 
                                         index: int, 
                                         spectra_index: bool = True
     ) -> int: # n_0,1
        """
        Get n_0,1 - the number of samples that the node did not participate in and were misclassified.

        Parameters:
        index (int): The index of the node.
        spectra_index (bool): Whether the index is the index of the node in the mapped tree or the index of the node in the spectra matrix.

        Returns:
        int: The number of samples that the node did not participate in and were misclassified.
        """
        
        if not spectra_index:
            index = self.mapped_tree.get_node(index).spectra_index
        
        return self.error_vector @ (1 - self.spectra[index])
    
    def get_accurate_nonparticipation_count(self, 
                                            index: int, 
                                            spectra_index: bool = True
     ) -> int: # n_0,0
        """
        Get n_0,0 - the number of samples that the node did not participate in and were classified correctly.

        Parameters:
        index (int): The index of the node.
        spectra_index (bool): Whether the index is the index of the node in the mapped tree or the index of the node in the spectra matrix.

        Returns:
        int: The number of samples that the node did not participate in and were classified correctly.
        """
        if not spectra_index:
            index = self.mapped_tree.get_node(index).spectra_index
        
        return (1 - self.error_vector) @ (1 - self.spectra[index])