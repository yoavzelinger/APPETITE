from .STAT import STAT
from .SFLDT import SFLDT, SFLDT_DEFAULT_SIMILARITY_MEASURES

class STAT_SFLDT:
    """
    The diagnoser that combines the STAT and SFLDT diagnosers.
    """
    def __init__(self, 
                 mapped_tree, 
                 X, 
                 y,
    ):
        """
        Initialize the diagnoser.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        similarity_measure (str): The similarity measure to use.
        """
        self.stat = STAT(mapped_tree, X, y)
        self.sfldt = SFLDT(mapped_tree, X, y, SFLDT_DEFAULT_SIMILARITY_MEASURES)
        self.diagnosis = None

    def get_diagnosis(self,
                      retrieve_scores=False
     ) -> list[int] | list[tuple[int, float]]:
        """
        Get the diagnosis.
        The diagnosis is calculated as the multiplication of the diagnosis of the STAT and SFLDT diagnosers.
        
        Returns:
        list[int] | list[tuple[int, float]]: The diagnosis. If retrieve_scores is True, the diagnosis will be a list of tuples with the node index and the score.
        """
        if self.diagnosis is None:
            stat_diagnosis = self.stat.get_diagnosis(retrieve_scores=True)
            sfldt_diagnosis = self.sfldt.get_diagnosis(retrieve_scores=True)
            self.diagnosis = [(node_index, stat_score * sfldt_score) for (node_index, stat_score), (_, sfldt_score) in zip(stat_diagnosis, sfldt_diagnosis)]
        if retrieve_scores:
            return self.diagnosis
        return [node_index for node_index, _ in self.diagnosis]
