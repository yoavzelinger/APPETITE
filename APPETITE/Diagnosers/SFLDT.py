from numpy import zeros, array as np_array

from .ADiagnoser import *
from APPETITE.Constants import SFLDT_DEFAULT_SIMILARITY_MEASURES

def get_faith_similarity(participation_vector: Series,
                         error_vector: Series
 ) -> float:
    """
    Get the faith similarity of the node to the error vector.

    The similarity is calculated by
        (error_participation +  0.5 * accurate_nonparticipation) /
        (error_participation + accurate_participation + error_nonparticipation + accurate_nonparticipation)
    Parameters:
        participation_vector (Series): The participation vector, where high value (1) represent high participation in the sample classification.
        error_vector (Series): The error vector, where high value (1) represent that the sample classified incorrectly.

    Returns:
        float: The faith similarity between the vectors
    """
    n11 = participation_vector @ error_vector
    n10 = participation_vector @ (1 - error_vector)
    n01 = (1 - participation_vector) @ error_vector
    n00 = (1 - participation_vector) @ (1 - error_vector)
    
    return (n11 +  0.5 * n00) / (n11 + n10 + n01 + n00)

class SFLDT(ADiagnoser):

    diagnoser_type = SINGLE_DIAGNOSER_TYPE_NAME

    similarity_measure_functions_dict = {
        "faith": get_faith_similarity
    }

    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series,
                 similarity_measure: str = SFLDT_DEFAULT_SIMILARITY_MEASURES
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
        self.node_count = mapped_tree.node_count
        self.sample_count = len(X)
        self.spectra = zeros((self.node_count, self.sample_count))
        self.error_vector = zeros(self.sample_count)
        self.similarity_measure = similarity_measure
        self.fill_spectra_and_error_vector(X, y)

    def update_fuzzy_participation(self,
     ) -> None:
        assert self.components_depths_vector.all()
        assert self.paths_depths_vector.all()

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
        self.components_depths_vector = np_array([self.mapped_tree.get_node(index=spectra_index, use_spectra_index=True).depth + 1 for spectra_index in range(self.node_count)])
        self.paths_depths_vector = zeros(self.sample_count)
        # Source: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#decision-path
        node_indicator = self.mapped_tree.sklearn_tree_model.tree_.decision_path(X.to_numpy(dtype="float32"))
        for sample_id in range(self.sample_count):
            participated_nodes = node_indicator.indices[
                node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
            ]
            for node in map(self.mapped_tree.get_node, participated_nodes):
                node_spectra_index = node.spectra_index
                self.spectra[node_spectra_index, sample_id] = 1
                if node.is_terminal():
                    self.paths_depths_vector[sample_id] = node.depth
                    error = node.class_name != y[sample_id]
                    self.error_vector[sample_id] = int(error)
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
            similarity_measure_function = SFLDT.similarity_measure_functions_dict[self.similarity_measure]
            self.diagnoses = [(spectra_index, similarity_measure_function(self.spectra[spectra_index], self.error_vector)) for spectra_index in range(self.node_count)]
            self.sort_diagnoses()
        diagnoses = self.get_diagnoses_with_return_indices(retrieve_spectra_indices)
        return super().get_diagnoses(retrieve_ranks, diagnoses)