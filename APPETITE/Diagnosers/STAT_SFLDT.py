from .ADiagnoser import *
from .STAT import STAT
from .SFLDT import SFLDT, SFLDT_DEFAULT_SIMILARITY_MEASURES

class STAT_SFLDT(ADiagnoser):
    """
    The diagnoser that combines the STAT and SFLDT diagnosers.
    """

    diagnoser_type = SINGLE_DIAGNOSER_TYPE_NAME

    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series,
                 similarity_measure: str = SFLDT_DEFAULT_SIMILARITY_MEASURES
    ):
        """
        Initialize the diagnoser.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        similarity_measure (str): The similarity measure to use.
        """
        super().__init__(mapped_tree, X, y)
        self.stat = STAT(mapped_tree, X, y)
        self.sfldt = SFLDT(mapped_tree, X, y, similarity_measure)

    def get_diagnoses(self,
                      retrieve_ranks=False
     ) -> list[int] | list[tuple[int, float]]:
        """
        Get the diagnoses.
        The diagnoses is calculated as the multiplication of the diagnoses of the STAT and SFLDT diagnosers.
        
        Returns:
        list[int] | list[tuple[int, float]]: The diagnoses. If retrieve_ranks is True, the diagnoses will be a list of tuples with the node index and the rank.
        """
        if self.diagnoses is None:
            stat_diagnoses = self.stat.get_diagnoses(retrieve_ranks=True)
            sfldt_diagnoses = self.sfldt.get_diagnoses(retrieve_ranks=True)
            multiple_ranks_dict = {}
            for single_diagnoses in (stat_diagnoses, sfldt_diagnoses):
                for node_index, rank in single_diagnoses:
                    multiple_ranks_dict[node_index] = multiple_ranks_dict.get(node_index, 1) * rank
            self.diagnoses = list(multiple_ranks_dict.items())
            self.sort_diagnoses()
        return super().get_diagnoses(retrieve_ranks)
