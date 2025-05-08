from numpy import zeros, array as np_array, max as np_max, ndarray, exp as np_exp, unique as np_unique, clip, mean as np_mean, log as np_log
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr as pearson_correlation

from sys import float_info
EPSILON = float_info.epsilon

from .ADiagnoser import *
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
    print("using faith")
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
    print("using correlation")
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
    print("using BCE")
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
        is_participation_binary = np_unique(participation_vector).size == 2
        is_error_binary = np_unique(error_vector).size == 2
        is_binaries = is_participation_binary, is_error_binary
        print(is_binaries)
        assert not all(is_binaries), "Only one of the vectors should be binary"
        assert any(is_binaries), "One of the vectors should be binary"
        if is_participation_binary:
            return participation_vector, error_vector
        return error_vector, participation_vector
    
    binary_vector, continuous_vector = get_binary_continuous_vectors(participation_vector, error_vector)
    continuous_vector = clip(continuous_vector, EPSILON, 1 - EPSILON)
    bce_loss = -np_mean(binary_vector * np_log(continuous_vector) + (1 - binary_vector) * np_log(1 - continuous_vector))
    return np_exp(-bce_loss)

def get_relevant_similarity_function(example_participation_vector1: ndarray,
                                     example_participation_vector2: ndarray,
                                     error_vector: ndarray
 ):
    """
    Get the relevant similarity function based of the type of the vectors:
    if both are binary - use faith similarity.
    if both are continuous - use correlation.
    if one is binary and the other continuous - use BCE similarity.
    Parameters:
        example_participation_vector1 (ndarray): The first participation vector.
        example_participation_vector2 (ndarray): The participation participation vector.
            * Using two participation vectors that aren't on the same path to avoid wrong function due to all samples classified using the same path.
        error_vector (ndarray): The error vector.
    
    Returns:
        The relevant similarity function
    """
    is_participation1_binary = np_unique(example_participation_vector1).size <= 2
    is_participation2_binary = np_unique(example_participation_vector2).size <= 2
    is_participation_binary = all((is_participation1_binary, is_participation2_binary))
    is_error_binary = np_unique(error_vector).size <= 2
    is_binaries = is_participation_binary, is_error_binary
    if all(is_binaries): # both binary
        return get_faith_similarity
    if any(is_binaries): # one binary one continuous
        return get_BCE_similarity
    # both continuous
    return get_correlation

class SFLDT(ADiagnoser):

    diagnoser_type = constants.SINGLE_DIAGNOSER_TYPE_NAME

    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series
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
        self.fill_spectra_and_error_vector(X, y)

    def update_fuzzy_participation(self,
     ) -> None:
        assert self.components_depths_vector.all()
        assert self.paths_depths_vector.all()
        if not constants.USE_FUZZY_PARTICIPATION:
            return
        self.components_depths_vector = self.components_depths_vector[:, None]
        self.spectra = (self.spectra * self.components_depths_vector) / self.paths_depths_vector
        assert np_max(self.spectra) <= 1.0, f"Participation should be in [0, 1] but got {np_max(self.spectra)}"

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
        self.components_depths_vector = np_array([self.mapped_tree.get_node(index=spectra_index, use_spectra_index=True).depth + 1 for spectra_index in range(self.components_count)])
        self.paths_depths_vector = zeros(self.tests_count)
        # Source: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#decision-path
        node_indicator = self.mapped_tree.sklearn_tree_model.tree_.decision_path(X.to_numpy(dtype="float32"))
        for sample_id in range(self.tests_count):
            participated_nodes = node_indicator.indices[
                node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
            ]
            for node in map(self.mapped_tree.get_node, participated_nodes):
                self.spectra[node.spectra_index, sample_id] = 1
                if node.is_terminal():
                    self.paths_depths_vector[sample_id] = node.depth + 1
                    self.error_vector[sample_id] = int(node.class_name != y[sample_id])
        self.update_fuzzy_participation()

    def get_diagnoses_with_return_indices(self,
                                          retrieve_spectra_indices: bool = False
    ):
        if retrieve_spectra_indices:
            return self.diagnoses
        convert_func = self.mapped_tree.convert_spectra_index_to_node_index
        returned_diagnoses = []
        for diagnosis, rank in self.diagnoses:
            if isinstance(diagnosis, int):
                diagnosis = convert_func(diagnosis)
            else:
                diagnosis = [convert_func(spectra_index) for spectra_index in diagnosis]
            returned_diagnoses.append((diagnosis, rank))
        return returned_diagnoses
    
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
            example_participation_vector1 = self.spectra[1]
            example_participation_vector2 = self.spectra[2]
            similarity_measure_function = get_relevant_similarity_function(example_participation_vector1, example_participation_vector2, self.error_vector)
            is_internal_node = lambda spectra_index: not self.mapped_tree.get_node(index=spectra_index, use_spectra_index=True).is_terminal()
            self.diagnoses = [(spectra_index, similarity_measure_function(self.spectra[spectra_index], self.error_vector) if is_internal_node(spectra_index) else 0) for spectra_index in range(self.components_count)]
            self.sort_diagnoses()
        diagnoses = self.get_diagnoses_with_return_indices(retrieve_spectra_indices)
        return super().get_diagnoses(retrieve_ranks, diagnoses)