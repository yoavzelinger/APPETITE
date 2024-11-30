# from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

from YoavNewCode.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree

class SFLDT:
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: pd.DataFrame,
                 y: pd.Series
    ):
        self.mapped_tree = mapped_tree
        self.node_count = mapped_tree.node_count
        self.sample_count = len(X)
        self.spectra = np.zeros((self.node_count, self.sample_count))
        self.error_vector = np.zeros(self.sample_count)

    def fill_spectra_and_error_vector(self, X, y):
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
    
    # TODO - VERIFY FUNCTIONS
    def get_errror_participation_count(self, 
                                       id: int, 
                                       spectra_id = True):
        if not spectra_id:
            id = self.mapped_tree.get_node(id).spectra_index
        
        return self.error_vector @ self.spectra[id]
    
    def get_error_nonparticipation_count(self, 
                                         id: int, 
                                         spectra_id = True):
        if not spectra_id:
            id = self.mapped_tree.get_node(id).spectra_index
        
        return self.error_vector @ (1 - self.spectra[id])
    
    def get_accurate_participation_count(self, 
                                         id: int, 
                                         spectra_id = True):
        if not spectra_id:
            id = self.mapped_tree.get_node(id).spectra_index
        
        return (1 - self.error_vector) @ self.spectra[id]
    
    def get_accurate_nonparticipation_count(self, 
                                            id: int, 
                                            spectra_id = True):
        if not spectra_id:
            id = self.mapped_tree.get_node(id).spectra_index
        
        return (1 - self.error_vector) @ (1 - self.spectra[id])