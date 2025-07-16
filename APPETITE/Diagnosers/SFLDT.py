from numpy import zeros, array as np_array, ndarray, exp as np_exp, clip, mean as np_mean, log as np_log, isclose as np_isclose, where as np_where, nan as np_nan
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr as pearson_correlation
from collections import defaultdict

from sys import float_info
EPSILON = float_info.epsilon

from .ADiagnoser import *
from .STAT import STAT
from APPETITE import Constants as constants

class SFLDT(ADiagnoser):
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series,
                 combine_stat: bool = constants.DEFAULT_COMBINE_STAT,
                 group_feature_nodes: bool = constants.DEFAULT_GROUP_FEATURE_NODES,
                 aggregate_tests: bool = constants.DEFAULT_AGGREGATE_TESTS_BY_PATHS,
                 combine_prior_confidence: bool = constants.DEFAULT_COMBINE_PRIOR_CONFIDENCE,
                 use_fuzzy_participation: bool = constants.DEFAULT_FUZZY_PARTICIPATION,
                 merge_singular_diagnoses: bool = constants.DEFAULT_MERGE_SINGULAR_DIAGNOSES    # DEPRECATED
    ):
        """
        Initialize the SFLDT diagnoser.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        combine_stat (bool): Whether to combine the diagnoses with the STAT diagnoser.
        use_fuzzy_participation (bool): Whether to use fuzzy components participation.
        aggregate_tests (bool): Whether to aggregate tests based on the classification paths.
        group_feature_nodes (bool): Whether to use feature components.
        combine_prior_confidence (bool): Whether to combine the confidence of the tests in the error vector calculation.
        
        # DEPRECATED #
           merge_singular_diagnoses (bool): Whether to merge singular diagnoses based on the features.
        """
        assert not (group_feature_nodes and merge_singular_diagnoses), "Cannot merge singular diagnoses with multiple fault diagnoser"
        super().__init__(mapped_tree, X, y)
        
        self.components_count = mapped_tree.node_count
        self.tests_count = len(X)
        
        self.spectra = zeros((self.components_count, self.tests_count))
        self.error_vector = zeros(self.tests_count)
        self.paths_depths_vector = zeros(self.tests_count)
        
        # Components
        self.group_feature_nodes = group_feature_nodes
        self.use_fuzzy_participation = use_fuzzy_participation
        self.explainer = TreeExplainer(mapped_tree.sklearn_tree_model, data=X.astype(np_float64), model_output="probability") if self.use_fuzzy_participation else None
        # Tests
        self.aggregate_tests = aggregate_tests
        self.combine_prior_confidence = combine_prior_confidence
        self.is_error_fuzzy = self.combine_prior_confidence or self.aggregate_tests
        # Deprecated TODO: Remove
        self.merge_singular_diagnoses = merge_singular_diagnoses

        self.fill_spectra_and_error_vector(X, y)
        self.stat = STAT(mapped_tree, X, y) if combine_stat else None

    def update_spectra_to_fuzzy(self) -> None:
        """
        Update the participation matrix to be fuzzy.
        Each participation will be calculated with a component factor normalized it's classification path depth.
        """
        components_factor = np_array([self.mapped_tree.get_node(index=spectra_index, use_spectra_index=True).depth + 1 for spectra_index in range(self.components_count)])[:, None]
        assert components_factor.all(), f"Components depths vector should be non-zero but got {components_factor}"
        self.spectra = (self.spectra * components_factor) / self.paths_depths_vector
        assert ((0 <= self.spectra) & (self.spectra <= 1)).all(), f"Components factor (numerator) vector should be less equal to paths depths (denominator - normalizer). Factor range: [{components_factor.min()}, {components_factor.max()}]; Paths depth: [{self.paths_depths_vector.min()}, {self.paths_depths_vector.max()}]."

    def aggregate_tests_by_paths(self
    ) -> None:
        """
        Merge the tests by their classification paths.
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

    def update_error_vector_to_fuzzy(self
    ) -> None:
        """
        Update all needed attributes to support the fuzzy error vector
        """
        if self.aggregate_tests:
            self.aggregate_tests_by_paths()

    def add_target_to_feature_components(self,
                                         target_name: str = "target"
    ) -> None:
        target_feature_index = len(self.feature_indices_dict)
        self.feature_indices_dict[target_name] = target_feature_index
        target_nodes = [node_spectra_index for node_spectra_index, node in self.mapped_tree.spectra_dict.items() if node.is_terminal()]
        self.feature_index_to_node_indices_dict[target_feature_index] = target_nodes

    def update_spectra_to_feature_components(self
    ) -> None:
        """
        Update the spectra matrix to be based on the features.
        Each feature will be represented by a single spectra index.
        The participation will be based on the average participation of all the nodes that represent the feature (which participated in the relevant test).
        """
        # Create feature to feature_index mapping (based on the order in the data)
        tree_columns = [feature for feature in self.X_after.columns if feature in self.mapped_tree.tree_features_set]
        self.feature_indices_dict = {feature: feature_index for (feature_index, feature) in enumerate(tree_columns)}
        # Create feature_index to the corresponding nodes mapping
        self.feature_index_to_node_indices_dict = defaultdict(list)
        for node_spectra_index, node in self.mapped_tree.spectra_dict.items():
            if node.is_terminal():
                continue
            self.feature_index_to_node_indices_dict[self.feature_indices_dict[node.feature]].append(node_spectra_index)
        # add target
        self.add_target_to_feature_components()
        self.components_count = len(self.feature_indices_dict)
        features_spectra = zeros((self.components_count, self.tests_count))
        for feature_index, feature_nodes_spectra_indices in self.feature_index_to_node_indices_dict.items(): #  here
            current_feature_spectra = self.spectra[feature_nodes_spectra_indices, :]
            current_feature_participations = np_where(current_feature_spectra > 0, current_feature_spectra, np_nan)
            features_spectra[feature_index] = np_mean(current_feature_participations, axis=0)
        self.spectra = features_spectra

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
            assert len(participated_nodes) >= 2, f"Test test_index: ({participated_nodes}, total {len(participated_nodes)}) has no participated nodes in the tree, but should have at least 2 (root and terminal node)."
            for node in map(self.mapped_tree.get_node, participated_nodes):
                self.spectra[node.spectra_index, test_index] = 1
                if node.is_terminal():
                    self.error_vector[test_index] = int(node.class_name != y[test_index])
                    if self.combine_prior_confidence:
                        self.error_vector[test_index] *= node.confidence
                    self.paths_depths_vector[test_index] = node.depth + 1
        assert self.paths_depths_vector.all(), f"Paths depths vector should be non-zero but got: \n{self.paths_depths_vector}"
        if self.is_error_fuzzy:
            self.update_error_vector_to_fuzzy()
        if self.use_fuzzy_participation:
            self.update_spectra_to_fuzzy()
        if self.group_feature_nodes:
            self.update_spectra_to_feature_components()
        
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
        for feature_index in features_diagnosis:
            nodes_diagnosis.extend(self.feature_index_to_node_indices_dict[feature_index])
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
        if self.group_feature_nodes:
            self.diagnoses = [(self.convert_features_diagnosis_to_nodes_diagnosis(diagnosis), rank) for diagnosis, rank in self.diagnoses]
        if retrieve_spectra_indices:
            return
        return_indices_diagnoses = []
        for diagnosis, rank in self.diagnoses:
            diagnosis = [self.mapped_tree.convert_spectra_index_to_node_index(spectra_index) for spectra_index in diagnosis]
            return_indices_diagnoses.append((diagnosis, rank))
        self.diagnoses = return_indices_diagnoses

# Similarity functions

    def get_faith_similarity(self, participation_vector: ndarray, error_vector: ndarray) -> float:
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

    def get_cosine_similarity(self, participation_vector: ndarray, error_vector: ndarray) -> float:
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

    def get_correlation(self, participation_vector: ndarray, error_vector: ndarray) -> float:
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
            return self.get_cosine_similarity(participation_vector, error_vector)
        return pearson_correlation(participation_vector, error_vector)[0]

    def get_BCE_similarity(self, participation_vector: ndarray, error_vector: ndarray) -> float:
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
            if self.is_error_fuzzy:
                return participation_vector, error_vector
            return error_vector, participation_vector
        
        binary_vector, continuous_vector = get_binary_continuous_vectors(participation_vector, error_vector)
        continuous_vector = clip(continuous_vector, EPSILON, 1 - EPSILON)
        bce_loss = -np_mean(binary_vector * np_log(continuous_vector) + (1 - binary_vector) * np_log(1 - continuous_vector))
        return np_exp(-bce_loss)
    
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
        are_continuous = self.use_fuzzy_participation, self.is_error_fuzzy
        if all(are_continuous): # both continuous
            if self.tests_count < 2:    # not enough samples for correlation measure
                return self.get_cosine_similarity
            return self.get_correlation
        if any(are_continuous): # one binary one continuous
            return self.get_BCE_similarity
        # both binary
        return self.get_faith_similarity
    
    def load_stat_diagnoses(self
     ) -> list[tuple[int, float]]:
        """
        Load the diagnoses from the STAT diagnoser.
        The diagnoses will be used to combine with the SFLDT diagnoses.
        """
        assert self.stat, "STAT diagnoser is not initialized"
        return self.stat.get_diagnoses(retrieve_ranks=True)
    
    def combine_stat_diagnoses(self
     ) -> None:
        """
        Combine stat diagnoses with the SFLDT diagnoses.
        the combination is done by multiplying the current diagnosis rank with the average STAT rank of all the corresponding diagnosis nodes.

        The process is as follows - for each diagnosis (can contains either single or multiple nodes):
        1. Convert the spectra indices for the corresponding node indices.
        2. Get the corresponding STAT ranks for the nodes.
        3. Calculate the average STAT rank for the nodes.
        4. Multiply the SFLDT rank with the calculated average STAT rank.
        """
        stat_diagnoses_dict = {node_index[0]: rank for node_index, rank in self.load_stat_diagnoses()}
        convert_spectra_to_node_indices_function = lambda spectra_indices: map(self.mapped_tree.convert_spectra_index_to_node_index, spectra_indices)
        get_nodes_stat_ranks_function = lambda spectra_indices: map(stat_diagnoses_dict.get, convert_spectra_to_node_indices_function(spectra_indices))
        get_average_stat_rank_function = lambda spectra_indices: max(0.5, sum(get_nodes_stat_ranks_function(spectra_indices)) / len(spectra_indices))
        get_nodes_from_features = lambda spectra_indices: self.convert_features_diagnosis_to_nodes_diagnosis(spectra_indices) if self.group_feature_nodes else spectra_indices
        self.diagnoses = [(diagnosis, sfldt_rank * get_average_stat_rank_function(get_nodes_from_features(diagnosis))) for diagnosis, sfldt_rank in self.diagnoses]

    def update_merge_singular_diagnoses(self
     ) -> None:
        """
        Merge singular diagnoses based on the features.
        The function will merge the diagnoses that have the same features.
        """
        merged_diagnoses = {}
        for diagnosis, rank in self.diagnoses:
            spectra_index = diagnosis[0]
            faulty_feature = self.mapped_tree.get_node(index=spectra_index, use_spectra_index=True).feature
            if faulty_feature is None:
                faulty_feature = "target"
            prior_feature_nodes, prior_rank = merged_diagnoses.get(faulty_feature, ([], -1))
            merged_diagnoses[faulty_feature] = (prior_feature_nodes + [spectra_index], max(prior_rank, rank))
        self.diagnoses = list(merged_diagnoses.values())
    
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
            if self.merge_singular_diagnoses:
                self.update_merge_singular_diagnoses()
        self.convert_diagnoses_indices(retrieve_spectra_indices)
        return super().get_diagnoses(retrieve_ranks)