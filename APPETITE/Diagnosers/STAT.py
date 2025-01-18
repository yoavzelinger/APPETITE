from copy import deepcopy

from .ADiagnoser import *

class STAT(ADiagnoser):
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series
    ):
        super().__init__(mapped_tree, X, y)

    @staticmethod
    def get_violation_ratio(node: MappedDecisionTree.DecisionTreeNode
     ) -> float:
        """
        Get the violation ratio of the node.

        Parameters:
        node (MappedDecisionTree.DecisionTreeNode): The node.

        Returns:
        float: The violation ratio of the node.
        """
        samples_reached_node_count = node.reached_samples_count
        if samples_reached_node_count == 0:
            return 0.0
        violated_samples_count = node.misclassifications_count if node.is_terminal() else node.left_child.reached_samples_count
        return 1.0 * violated_samples_count / samples_reached_node_count

    def get_before_violation(self,
                             node_index: int
     ) -> float:
        """
        Get the violation of the node before the drift.
        
        Parameters:
        node_index (int): The node index.
        
        Returns:
        float: The violation of the node before the drift.
        """
        node = self.mapped_tree.get_node(node_index)
        return STAT.get_violation_ratio(node)
    
    def get_after_violation(self,
                            node_index: int
     ) -> float:
        """
        Get the violation of the node after the drift.
        
        Parameters:
        node_index (int): The node index.
        
        Returns:
        float: The violation of the node after the drift.
        """
        node = self.mapped_tree.get_node(node_index)
        node_after = deepcopy(node)
        node_after.update_node_data_attributes(self.X_after, self.y_after)
        if not node.is_terminal():
            node_after.left_child = deepcopy(node.left_child)
            node_after.left_child.update_node_data_attributes(self.X_after, self.y_after)
        return STAT.get_violation_ratio(node_after)
    
    def get_node_violation_difference(self,
                                      node_index: int,
     ) -> float:
        """
        Get the violation difference of the node.
        
        Parameters:
        node_index (int): The node index.
        
        Returns:
        float: The violation difference of the node.
        """
        before_violation = self.get_before_violation(node_index)
        after_violation = self.get_after_violation(node_index)
        return abs(before_violation - after_violation)

    def get_diagnoses(self,
                      retrieve_ranks: bool = False
     ) -> list[int] | list[tuple[int, float]]:
        """
        Get the diagnoses of the drift.


        Parameters:
        retrieve_ranks (bool): Whether to return the ranks of the nodes.
        
        Returns:
        list[int] | list[tuple[int, float]]: The diagnosis. If retrieve_ranks is True, the diagnosis will be a list of tuples,
          where the first element is the index and the second is the violation ratio.
        """
        if self.diagnoses is None:
            self.diagnoses = [(node_index, self.get_node_violation_difference(node_index)) for node_index in self.mapped_tree.tree_dict.keys()]
            self.sort_diagnoses()
        return super().get_diagnoses(retrieve_ranks)