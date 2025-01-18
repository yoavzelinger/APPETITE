from .ADiagnoser import *
from .STAT import STAT
from .BARINEL import BARINEL

class STAT_BARINEL(ADiagnoser):
    """
    The diagnoser that combines the STAT and BARINEL diagnosers.
    """
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
        self.stat = STAT(mapped_tree, X, y)
        self.barinel = BARINEL(mapped_tree, X, y)
    
    def get_stat_diagnoses(self,
                           diagnosis_type: str = 'difference',   # | 'before' | 'after'
     ) -> list[tuple[int, float]]:
        """
        Get the STAT diagnoses.

        Parameters:
        diagnosis_type (str): The type of diagnosis. It can be 'difference', 'before', or 'after'.

        Returns:
        list[tuple[int, float]]: The STAT diagnoses with their ranks.
        """
        assert diagnosis_type in ['difference', 'before', 'after']
        if diagnosis_type == 'difference':
            return [(node_index, self.stat.get_node_violation_difference(node_index)) for node_index in self.mapped_tree.get_node_indices()]
        elif diagnosis_type == 'before':
            return [(node_index, self.stat.get_before_violation(node_index)) for node_index in self.mapped_tree.get_node_indices()]
        return [(node_index, self.stat.get_after_violation(node_index)) for node_index in self.mapped_tree.get_node_indices()]

    def get_diagnoses(self,
                      retrieve_ranks: bool = False,
                      retrieve_spectra_indices: bool = False,
     ) -> list[int] | list[tuple[int, float]]:
        """
        Get the diagnoses.
        For the prior probability to the BARINEL diagnoses, we use the average STAT of all the nodes in each diagnosis.
        Thus, the diagnoses is calculated as the multiplication of the diagnoses of the STAT and BARINEL diagnosers.

        
        Returns:
        list[int] | list[tuple[int, float]]: The diagnoses. If retrieve_ranks is True, the diagnoses will be a list of tuples with the node index and the rank.
        """
        if self.diagnoses is None:
            self.diagnoses = self.barinel.get_diagnoses(retrieve_ranks=True, retrieve_spectra_indices=retrieve_spectra_indices)
            for diagnosis_index, (diagnosis, barinel_rank) in enumerate(self.diagnoses):
                stat_rank = 1
                for node_index in diagnosis:
                    if retrieve_spectra_indices:
                        node_index = self.mapped_tree.convert_spectra_index_to_node_index(node_index)
                    stat_rank *= self.stat.get_after_violation(node_index)
                self.diagnoses[diagnosis_index] = (diagnosis, barinel_rank * stat_rank)
            self.sort_diagnoses()
        return super().get_diagnoses(retrieve_ranks)