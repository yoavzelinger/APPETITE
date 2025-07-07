from numpy import zeros, clip, newaxis
from pandas import DataFrame, Series

from APPETITE import Constants as constants
from APPETITE.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree
from .SFLDT import SFLDT
from .BARINEL_Paths import BARINEL_Paths
from .BARINEL_Paths_After import BARINEL_Paths_After
from .BARINEL_Paths_Difference import BARINEL_Paths_Difference

class BARINEL_Features(BARINEL_Paths):
    
    diagnoser_type = constants.MULTIPLE_DIAGNOSER_TYPE_NAME

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
    
    def fill_spectra_and_error_vector(self, 
                                      X: DataFrame, 
                                      y: Series,
                                      use_fuzzy_error: bool = constants.DEFAULT_FUZZY_ERROR
     ) -> None:
        if use_fuzzy_error:
            return BARINEL_Paths.fill_spectra_and_error_vector(self, X, y)
        return SFLDT.fill_spectra_and_error_vector(self, X, y)
    
    def update_fuzzy_participation(self) -> None:
        if constants.DEFAULT_FUZZY_PARTICIPATION:
            if len(self.spectra) == len(self.components_depths_vector):
                self.spectra = self.spectra / self.components_depths_vector[:, newaxis]
        else:
            self.spectra = clip(self.spectra, 0, 1)
        super().update_fuzzy_participation()

    def get_fuzzy_error_data(self,
                       before_accuracy_vector: Series,
                       current_accuracy_vector: Series,
                       barinel_paths_type: str
    ) -> tuple[Series, float, float]:
        if barinel_paths_type == "AFTER":
            return BARINEL_Paths_After.get_fuzzy_error_data(self, before_accuracy_vector, current_accuracy_vector)
        elif barinel_paths_type == "DIFFERENCE":
            return BARINEL_Paths_Difference.get_fuzzy_error_data(self, before_accuracy_vector, current_accuracy_vector)
        raise ValueError(f"Unknown BARINEL_PATHS_TYPE: {barinel_paths_type}. Use 'AFTER' or 'DIFFERENCE'.")

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
                      retrieve_spectra_indices: bool = False,
     ) -> list[list[int]] | list[tuple[list[int], float]]:
        BARINEL_Paths.get_diagnoses(self, retrieve_ranks=True, retrieve_spectra_indices=True)
        for diagnosis_index, (features_diagnosis, rank) in enumerate(self.diagnoses):
            if isinstance(features_diagnosis, int):
                features_diagnosis = [features_diagnosis]
            nodes_diagnosis = self.convert_features_diagnosis_to_nodes_diagnosis(features_diagnosis)
            self.diagnoses[diagnosis_index] = (nodes_diagnosis, rank)
        return super().get_diagnoses(retrieve_ranks, retrieve_spectra_indices)