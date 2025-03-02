from .ADiagnoser import *
from .BARINEL import BARINEL

class BARINEL_Combo(ADiagnoser):
    """
    The diagnoser that combines the STAT and BARINEL diagnosers.
    """

    diagnoser_type = MULTIPLE_DIAGNOSER_NAME
    
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series
    ):
        """
        Initialize the diagnoser.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        """
        super().__init__(mapped_tree, X, y)
        self.barinel = BARINEL(mapped_tree, X, y)
    
    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
                      retrieve_spectra_indices: bool = False,
     ) -> list[int] | list[tuple[int, float]]:
        """
        Get the diagnoses.
        The diagnosis ranks are calculated by 

        
        Returns:
        list[int] | list[tuple[int, float]]: The diagnoses. If retrieve_ranks is True, the diagnoses will be a list of tuples with the node index and the rank.
        """
        if self.diagnoses is None:
            barinel_diagnoses = self.barinel.get_diagnoses(retrieve_ranks=True, retrieve_spectra_indices=retrieve_spectra_indices)
            nodes_cumulative_diagnoses = {}
            for diagnosis, rank in barinel_diagnoses:
                for node_index in diagnosis:
                    node_ranks, node_sums = nodes_cumulative_diagnoses.get(node_index, (0, 0))
                    nodes_cumulative_diagnoses[node_index] = (node_ranks + rank, node_sums + 1)

            self.diagnoses = []
            for diagnosis, _ in barinel_diagnoses:
                nodes_ranks_sum, nodes_diagnoses_count = 0, 0
                for node_index in diagnosis:
                    nodes_ranks_sum = nodes_cumulative_diagnoses[node_index][0]
                    nodes_diagnoses_count = nodes_cumulative_diagnoses[node_index][1]
                self.diagnoses.append((diagnosis, nodes_ranks_sum / nodes_diagnoses_count))
            self.sort_diagnoses()
        return super().get_diagnoses(retrieve_ranks)