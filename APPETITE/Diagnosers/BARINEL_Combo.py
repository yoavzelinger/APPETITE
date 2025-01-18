from .ADiagnoser import *
from .BARINEL import BARINEL

class BARINEL_Combo(ADiagnoser):
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
            features_cumulative_diagnoses = {feature: (0, 0) for feature in self.mapped_tree.tree_features_set}
            for diagnosis, rank in barinel_diagnoses:
                for diagnosis_faulty_feature in map(lambda node_index: self.mapped_tree.get_node(node_index, use_spectra_index=retrieve_spectra_indices).get_feature_extended(), diagnosis):
                    features_cumulative_diagnoses[diagnosis_faulty_feature] = (features_cumulative_diagnoses[diagnosis_faulty_feature][0] + rank, features_cumulative_diagnoses[diagnosis_faulty_feature][1] + 1)

            self.diagnoses = []
            for diagnosis, _ in barinel_diagnoses:
                features_ranks_sum, features_diagnoses_count = 0, 0
                for diagnosis_faulty_feature in map(lambda node_index: self.mapped_tree.get_node(node_index, use_spectra_index=retrieve_spectra_indices).get_feature_extended(), diagnosis):
                    features_ranks_sum += features_cumulative_diagnoses[diagnosis_faulty_feature][0]
                    features_diagnoses_count += features_cumulative_diagnoses[diagnosis_faulty_feature][1]
                self.diagnoses.append((diagnosis, features_ranks_sum / features_diagnoses_count))
            self.sort_diagnoses()
        return super().get_diagnoses(retrieve_ranks)