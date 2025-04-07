from numpy import zeros, clip, array as np_array
from pandas import DataFrame, Series

from APPETITE.Constants import BARINEL_STD_THRESHOLD
from APPETITE.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree
from .BARINEL import BARINEL

from .BARINEL_Features import BARINEL_Features
from .STAT_SFLDT import STAT_SFLDT

class BARINEL_Features_STAT_SFLDT(BARINEL_Features):
    def __init__(self,
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series,
    ) -> None:
        super().__init__(mapped_tree, X, y)
        self.single_diagnoser = STAT_SFLDT(mapped_tree, X, y)

    def convert_features_diagnosis_to_nodes_diagnosis(self,
            features_diagnosis: list[int]
            ) -> list[int]:
        nodes_rank_dict = {}
        for node_index, node_rank in self.single_diagnoser.get_diagnoses(retrieve_ranks=True):
            spectra_index = self.mapped_tree.convert_node_index_to_spectra_index(node_index)
            nodes_rank_dict[spectra_index] = node_rank
        nodes_diagnosis = []
        for feature_spectra_index in features_diagnosis:
            feature = self.spectra_features_dict[feature_spectra_index]
            feature_nodes_spectra_indices = self.features_spectra_dict[feature][1]
            feature_nodes_ranks = [nodes_rank_dict[node_spectra_index] for node_spectra_index in feature_nodes_spectra_indices]
            feature_nodes_ranks = np_array(feature_nodes_ranks)
            feature_nodes_rank_average, feature_nodes_rank_std = feature_nodes_ranks.mean(), feature_nodes_ranks.std()
            threshold = feature_nodes_rank_average + BARINEL_STD_THRESHOLD * feature_nodes_rank_std
            threshold = min(threshold, max(feature_nodes_ranks))
            feature_relevant_nodes = [node_spectra_index for node_spectra_index in feature_nodes_spectra_indices if nodes_rank_dict[node_spectra_index] >= threshold]
            nodes_diagnosis.extend(feature_relevant_nodes)
        return nodes_diagnosis

    
