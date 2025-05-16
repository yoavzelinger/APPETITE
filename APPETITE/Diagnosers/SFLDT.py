from numpy import zeros, array as np_array, max as np_max, ndarray, exp as np_exp, clip, mean as np_mean, log as np_log, isclose as np_isclose
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr as pearson_correlation
from collections import defaultdict

from sys import float_info
EPSILON = float_info.epsilon

from .ADiagnoser import *
from .STAT import STAT
from APPETITE import Constants as constants

def get_faith_similarity(participation_vector: ndarray,
                         error_vector: ndarray
 ) -> float:
    """
    Get the faith similarity of the component to the error vector.

    The similarity is calculated by
        (error_participation +  0.5 * accurate_nonparticipation) /
        (error_participation + accurate_participation + error_nonparticipation + accurate_nonparticipation)
    Parameters:
        participation_vector (ndarray): The participation vector, where high value (1) represent high participation in the sample classification.
        error_vector (ndarray): The error vector, where high value (1) represent that the sample classified incorrectly.

    Returns:
        float: The faith similarity between the vectors.
    """
    n11 = participation_vector @ error_vector
    n10 = participation_vector @ (1 - error_vector)
    n01 = (1 - participation_vector) @ error_vector
    n00 = (1 - participation_vector) @ (1 - error_vector)
    
    return (n11 +  0.5 * n00) / (n11 + n10 + n01 + n00)

def get_cosine_similarity(participation_vector: ndarray,
                          error_vector: ndarray
 ) -> float:
    """
    Get the cosine similarity of the component to the error vector.
    Parameters:
        participation_vector (ndarray): The participation vector, where high value (1) represent high participation in the sample classification.
        error_vector (ndarray): The error vector, where high value (1) represent that the sample classified incorrectly.

    Returns:
        float: The cosine similarity between the vectors.
    """
    participation_vector, error_vector = participation_vector[None, :], error_vector[None, :]
    return cosine_similarity(participation_vector, error_vector)[0][0]

def get_correlation(participation_vector: ndarray,
                    error_vector: ndarray
 ) -> float:
    """
    Get the correlation of the component to the error vector.
    Parameters:
        participation_vector (ndarray): The participation vector, where high value (1) represent high participation in the sample classification.
        error_vector (ndarray): The error vector, where high value (1) represent that the sample classified incorrectly.

    Returns:
        float: The correlation similarity between the vectors.
    """
    # check if the participation vector is all 0
    if all(np_isclose(participation_vector, participation_vector[0])):  # constant participation, cannot calculate correlation, using cosine similarity instead
        return get_cosine_similarity(participation_vector, error_vector)
    return pearson_correlation(participation_vector, error_vector)[0]

def get_BCE_similarity(participation_vector: ndarray,
                       error_vector: ndarray
 ) -> float:
    """
    Get binary-cross-entropy similarity between the two vectors.
    for this similarity one of the vectors should be binary.
    the similarity is calculated as e^(-BCE) so high value means strong relationship.
    Parameters:
        participation_vector (ndarray): The participation vector, where high value (1) represent high participation in the sample classification.
        error_vector (ndarray): The error vector, where high value (1) represent that the sample classified incorrectly.

    Returns:
        float: The binary-cross-entropy similarity between the vectors.
    """
    def get_binary_continuous_vectors(participation_vector: ndarray,
                                      error_vector: ndarray
     ) -> tuple[ndarray]:
        """
        determine which vector is binary and which is continuous.
        Parameters:
            participation_vector (ndarray): The participation vector, where high value (1) represent high participation in the sample classification.
            error_vector (ndarray): The error vector, where high value (1) represent that the sample classified incorrectly.
        
        Returns:
            ndarray: the binary vector.
            ndarray: the continuous vector.
        """
        if constants.DEFAULT_FUZZY_PARTICIPATION:
            return participation_vector, error_vector
        return error_vector, participation_vector
    
    binary_vector, continuous_vector = get_binary_continuous_vectors(participation_vector, error_vector)
    continuous_vector = clip(continuous_vector, EPSILON, 1 - EPSILON)
    bce_loss = -np_mean(binary_vector * np_log(continuous_vector) + (1 - binary_vector) * np_log(1 - continuous_vector))
    return np_exp(-bce_loss)

class SFLDT(ADiagnoser):
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series,
                 combine_stat: bool = constants.DEFAULT_ADD_STAT,
                 use_fuzzy_participation: bool = constants.DEFAULT_FUZZY_PARTICIPATION,
                 use_fuzzy_error: bool = constants.DEFAULT_FUZZY_ERROR,
                 use_feature_components: bool = constants.DEFAULT_FEATURE_COMPONENTS
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
        self.components_count = mapped_tree.node_count
        self.tests_count = len(X)
        self.spectra = zeros((self.components_count, self.tests_count))
        self.error_vector = zeros(self.tests_count)
        self.paths_depths_vector = zeros(self.tests_count)
        self.use_fuzzy_participation, self.use_fuzzy_error = use_fuzzy_participation, use_fuzzy_error
        self.use_feature_components = use_feature_components
        self.fill_spectra_and_error_vector(X, y)
        self.stat = STAT(mapped_tree, X, y) if combine_stat else None

    def update_fuzzy_participation(self,
                                   components_factor: ndarray = None
    ) -> None:
        """
        Update the participation matrix to be fuzzy.
        Each participation will be calculated with a component factor normalized it's classification path depth.
        """
        if components_factor is None:
            components_factor = np_array([self.mapped_tree.get_node(index=spectra_index, use_spectra_index=True).depth + 1 for spectra_index in range(self.components_count)])[:, None]
            assert components_factor.all(), f"Components depths vector should be non-zero but got {components_factor}"
        self.spectra = (self.spectra * components_factor) / self.paths_depths_vector
        assert np_max(self.spectra) <= 1.0, f"Participation should be in [0, 1] but got {np_max(self.spectra)}"

    def update_fuzzy_error(self
    ) -> None:
        """
        Update the error vector to be fuzzy.
        Each test will be correspond to classification path in the tree (that has any nodes passed through).
        The error value will be the average error (the misclassification) of the classification path.
        """
        path_tests_indices = defaultdict(list)
        for test_index in range(self.tests_count):
            test_participation_vector = tuple(self.spectra[:, test_index])
            path_tests_indices[test_participation_vector].append(test_index)

        self.spectra = np_array(list(path_tests_indices.keys())).T
        self.error_vector = np_array([self.error_vector[test_indices].mean() for test_indices in path_tests_indices.values()])
        self.paths_depths_vector = np_array([self.paths_depths_vector[test_indices].min() for test_indices in path_tests_indices.values()])
        self.tests_count = self.error_vector.shape[0]

    def update_feature_components(self
    ) -> None:
        """
        Update the spectra matrix to be based on the features.
        Each feature will be represented by a single spectra index.
        If fuzzy participation is used, each component (feature) factor will be the amount of nodes of the corresponding feature in the relevant classification path.
        """
        self.feature_to_spectra_dict = {}  # {feature: feature_spectra_index, [node_spectra_indices]}
        self.spectra_index_to_features_dict = {}  # {feature_spectra_index: feature}
        for node_spectra_index, node in self.mapped_tree.spectra_dict.items():
            if node.is_terminal():
                continue
            if node.feature not in self.feature_to_spectra_dict:
                feature_spectra_index = len(self.feature_to_spectra_dict)
                self.feature_to_spectra_dict[node.feature] = feature_spectra_index, []
                self.spectra_index_to_features_dict[feature_spectra_index] = node.feature
            self.feature_to_spectra_dict[node.feature][1].append(node_spectra_index)
        self.components_count = len(self.feature_to_spectra_dict)
        features_spectra = zeros((self.components_count, self.spectra.shape[1]))
        features_count_vector = zeros((self.components_count, self.tests_count))
        for feature_spectra_index, feature_nodes_spectra_indices in self.feature_to_spectra_dict.values():
            features_spectra[feature_spectra_index] = 1
            features_count_vector[feature_spectra_index] = self.spectra[feature_nodes_spectra_indices, :].sum(axis=0)
        self.spectra = features_spectra
        if self.use_fuzzy_participation:
            self.update_fuzzy_participation(components_factor=features_count_vector)

    def fill_spectra_and_error_vector(self, 
                                      X: DataFrame, 
                                      y: Series
    ) -> None:
        """
        Fill the spectra matrix and the error vector.

        Parameters:
        X (DataFrame): The data.
        y (Series): The target column.
        """
        # Source: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#decision-path
        node_indicator = self.mapped_tree.sklearn_tree_model.tree_.decision_path(X.to_numpy(dtype="float32"))
        for test_index in range(self.tests_count):
            participated_nodes = node_indicator.indices[
                node_indicator.indptr[test_index] : node_indicator.indptr[test_index + 1]
            ]
            for node in map(self.mapped_tree.get_node, participated_nodes):
                self.spectra[node.spectra_index, test_index] = 1
                if node.is_terminal():
                    self.error_vector[test_index] = int(node.class_name != y[test_index])
                    self.paths_depths_vector[test_index] = node.depth + 1
        assert self.paths_depths_vector.all(), f"Paths depths vector should be non-zero but got {self.paths_depths_vector}"
        if self.use_fuzzy_error:
            self.update_fuzzy_error()
        if self.use_feature_components:
            self.update_feature_components()
        elif self.use_fuzzy_participation:
            self.update_fuzzy_participation()
        
    def convert_features_diagnosis_to_nodes_diagnosis(self,
                                                      features_diagnosis: list[int]
     ) -> list[int]:
        """
        Convert the features diagnosis to nodes diagnosis.
        The function will return the spectra indices of the nodes that their feature are part of the features diagnosis.
        Parameters:
            features_diagnosis (list[int]): The features diagnosis.
        Returns:
            list[int]: The diagnosis with node spectra indices.
        """
        nodes_diagnosis = []
        for feature_spectra_index in features_diagnosis:
            feature = self.spectra_index_to_features_dict[feature_spectra_index]
            nodes_diagnosis.extend(self.feature_to_spectra_dict[feature][1])
        return nodes_diagnosis
    
    def convert_diagnoses_indices(self,
                                  retrieve_spectra_indices: bool = False
    ) -> None:
        """
        Convert the diagnoses indices from spectra indices to node indices.
        If the feature components are used, the function will first convert the indices to node indices.
        Parameters:
            retrieve_spectra_indices (bool): Whether to return the spectra indices or the node indices.
        """
        if self.use_feature_components:
            self.diagnoses = [(self.convert_features_diagnosis_to_nodes_diagnosis(diagnosis), rank) for diagnosis, rank in self.diagnoses]
        if retrieve_spectra_indices:
            return
        return_indices_diagnoses = []
        for diagnosis, rank in self.diagnoses:
            diagnosis = [self.mapped_tree.convert_spectra_index_to_node_index(spectra_index) for spectra_index in diagnosis]
            return_indices_diagnoses.append((diagnosis, rank))
        self.diagnoses = return_indices_diagnoses
    
    def get_relevant_similarity_function(self):
        """
        Get the relevant similarity function based of the type of the vectors (participation and error).:
        if both are binary - use faith similarity.
        if both are continuous - use correlation.
        if one is binary and the other continuous - use BCE similarity.
        Parameters:
        
        Returns:
            The relevant similarity function
        """
        are_continuous = self.use_fuzzy_participation, self.use_fuzzy_error
        if all(are_continuous): # both continuous
            if self.tests_count < 2:    # not enough samples for correlation measure
                return get_cosine_similarity
            return get_correlation
        if any(are_continuous): # one binary one continuous
            return get_BCE_similarity
        # both binary
        return get_faith_similarity
    
    def combine_stat_diagnoses(self
     ) -> None:
        """
        Combine stat diagnoses with the SFLDT diagnoses.
        the combination is done by multiplying the current diagnosis rank with the average STAT rank of all the corresponding diagnosis nodes.
        """
        self.stat_diagnoses = self.stat.get_diagnoses(retrieve_ranks=True)
        self.stat_diagnoses_dict = {node_index[0]: rank for node_index, rank in self.stat_diagnoses}
        if self.diagnoses is None: # no diagnoses to combine with
            return
        convert_spectra_to_node_indices_function = lambda spectra_indices: map(self.mapped_tree.convert_spectra_index_to_node_index, spectra_indices)
        get_average_stat_rank_function = lambda spectra_indices: sum(convert_spectra_to_node_indices_function(spectra_indices)) / len(spectra_indices)
        self.diagnoses = [(diagnosis, sfldt_rank * get_average_stat_rank_function(diagnosis)) for diagnosis, sfldt_rank in self.diagnoses]
    
    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
                      retrieve_spectra_indices: bool = False
     ) -> list[int] | list[tuple[int, float]]:
        """
        Get the diagnoses of the nodes.
        The diagnoses consists the nodes ordered by their similarity to the error vector (DESC).

        Parameters:
        retrieve_spectra_indices (bool): Whether to return the spectra indices or the node indices.
        retrieve_ranks (bool): Whether to return the diagnoses rank.

        Returns:
        list[int] | list[tuple[int, float]]: The diagnoses. If retrieve_ranks is True, the diagnoses will be a list of tuples,
          where the first element is the index and the second is the similarity rank.
        """
        if self.diagnoses is None:
            similarity_measure_function = self.get_relevant_similarity_function()
            self.diagnoses = [([spectra_index], similarity_measure_function(self.spectra[spectra_index], self.error_vector)) for spectra_index in range(self.components_count)]
            if self.stat:
                self.combine_stat_diagnoses()
            self.convert_diagnoses_indices(retrieve_spectra_indices)
        return super().get_diagnoses(retrieve_ranks)