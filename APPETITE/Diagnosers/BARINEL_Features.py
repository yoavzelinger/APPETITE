from numpy import zeros, clip
from pandas import DataFrame, Series

from APPETITE.Constants import MULTIPLE_DIAGNOSER_TYPE_NAME
from APPETITE.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree
from .SFLDT import SFLDT
from .FuzzySFLDT import FuzzySFLDT
from .BARINEL import BARINEL
from .BARINEL_Paths_After import BARINEL_Paths_After
from .BARINEL_Paths_Difference import BARINEL_Paths_Difference

class BARINEL_Features(BARINEL):
    
    diagnoser_type = MULTIPLE_DIAGNOSER_TYPE_NAME

    def __init__(self,
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series,
    ) -> None:
        super().__init__(mapped_tree, X, y)
        self.features_spectra_dict = {}  # {feature: feature_spectra_index, [node_spectra_indices]}
        self.spectra_features_dict = {}  # {feature_spectra_index: feature}
        for node_spectra_index, node in self.mapped_tree.spectra_dict.items():
            feature = node.feature if not node.is_terminal() else "target"
            if feature == "target":
                continue
            if feature not in self.features_spectra_dict:
                feature_spectra_index = len(self.features_spectra_dict)
                self.features_spectra_dict[feature] = feature_spectra_index, []
                self.spectra_features_dict[feature_spectra_index] = feature
            self.features_spectra_dict[feature][1].append(node_spectra_index)
        self.components_count = len(self.features_spectra_dict)
        spectra = zeros((self.components_count, self.spectra.shape[1]))
        for feature_spectra_index, feature_nodes_spectra_indices in self.features_spectra_dict.values():
            spectra[feature_spectra_index] = self.spectra[feature_nodes_spectra_indices, :].sum(axis=0)
        self.spectra = spectra
    
    def update_fuzzy_participation(self) -> None:
        if isinstance(self, SFLDT):
            self.spectra = clip(self.spectra, 0, 1)
        elif isinstance(self, FuzzySFLDT) and len(self.spectra) == len(self.components_depths_vector):
            self.spectra = self.spectra / self.components_depths_vector
        super().update_fuzzy_participation()

    def convert_features_diagnosis_to_nodes_diagnosis(self,
            features_diagnosis: list[int]
            ) -> list[int]:
        nodes_diagnosis = []
        for feature_spectra_index in features_diagnosis:
            feature = self.spectra_features_dict[feature_spectra_index]
            nodes_diagnosis.extend(self.features_spectra_dict[feature][1])
        return nodes_diagnosis

    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
                      retrieve_spectra_indices: bool = False
     ) -> list[list[int]] | list[tuple[list[int], float]]:
        super().get_diagnoses(retrieve_ranks=True, retrieve_spectra_indices=True)
        for diagnosis_index, (features_diagnosis, rank) in enumerate(self.diagnoses):
            if isinstance(features_diagnosis, int):
                features_diagnosis = [features_diagnosis]
            nodes_diagnosis = self.convert_features_diagnosis_to_nodes_diagnosis(features_diagnosis)
            self.diagnoses[diagnosis_index] = (nodes_diagnosis, rank)
        return super().get_diagnoses(retrieve_ranks, retrieve_spectra_indices)