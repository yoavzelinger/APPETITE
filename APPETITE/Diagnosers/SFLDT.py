from pandas import DataFrame, Series
from numpy import zeros

from APPETITE.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree

from APPETITE.Constants import SFLDT_DEFAULT_SIMILARITY_MEASURES

def get_faith_similarity(participation_vector: Series,
                         error_vector: Series
 ) -> float:
    """
    Get the faith similarity of the node to the error vector.

    The similarity is calculated by
        (errror_participation +  0.5 * accurate_nonparticipation) /
        (errror_participation + accurate_participation + error_nonparticipation + accurate_nonparticipation)
    Parameters:
        participation_vector (Series): The participation vector, where 1 represent participation in the sample classification.
        error_vector (Series): The error vector, where 1 represent that the sample classified incorrectly.

    Returns:
        float: The faith similarity between the vectors
    """
    n11 = participation_vector @ error_vector
    n10 = participation_vector @ (1 - error_vector)
    n01 = (1 - participation_vector) @ error_vector
    n00 = (1 - participation_vector) @ (1 - error_vector)
    
    return (n11 +  0.5 * n00) / (n11 + n10 + n01 + n00)

class SFLDT:

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
        self.mapped_tree = mapped_tree
        self.node_count = mapped_tree.node_count
        self.sample_count = len(X)
        self.spectra = zeros((self.node_count, self.sample_count))
        self.error_vector = zeros(self.sample_count)
        self.similarity_measure = similarity_measure
        self.fill_spectra_and_error_vector(X, y)

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
                      retrieve_spectra_indices: bool = False,
                      retrive_scores: bool = False
     ) -> list[int] | list[tuple[int, float]]:
        """
        Get the diagnosis of the nodes.
        The diagnosis consists the nodes ordered by their similarity to the error vector (DESC).

        Parameters:
        retrieve_spectra_indices (bool): Whether to return the spectra indices or the node indices.

        Returns:
        list[int] | list[tuple[int, float]]: The diagnosis. If retrive_scores is True, the diagnosis will be a list of tuples,
          where the first element is the index and the second is the similarity score.
        """
        similarity_measure_function = SFLDT.similarity_measure_functions_dict[self.similarity_measure]
        get_returned_index = lambda spectra_index: spectra_index if retrieve_spectra_indices else self.mapped_tree.convert_spectra_index_to_node_index(spectra_index)
        index_rank = [(get_returned_index(spectra_index), similarity_measure_function(self.spectra[spectra_index], self.error_vector)) for spectra_index in range(self.node_count)]
        index_rank.sort(key=lambda x: x[1], reverse=True)
        if retrive_scores:
            return index_rank
        return [index for index, _ in index_rank]