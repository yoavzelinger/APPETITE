from pandas import DataFrame, Series
from copy import deepcopy

from APPETITE.DecisionTreeTools.MappedDecisionTree import MappedDecisionTree

class STAT:
    def __init__(self, 
                 mapped_tree: MappedDecisionTree,
                 X: DataFrame,
                 y: Series
    ):
        """
        Initialize the STAT diagnoser.
        
        Parameters:
        mapped_tree (MappedDecisionTree): The mapped decision tree.
        X (DataFrame): The data.
        y (Series): The target column.
        """
        self.mapped_tree = mapped_tree
        self.X_after = X
        self.y_after = y
        self.diagnosis = None

    def get_violation_ratio(self,
                            node: MappedDecisionTree.DecisionTreeNode
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
        return self.get_violation_ratio(node)
    
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
        return self.get_violation_ratio(node_after)
    
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

    def get_diagnosis(self,
                      retrive_scores: bool = False
     ) -> list[int] | list[tuple[int, float]]:
        """
        Get the diagnosis of the drift.

        Parameters:
        retrive_scores (bool): Whether to return the scores of the nodes.
        
        Returns:
        list[int] | list[tuple[int, float]]: The diagnosis. If retrive_scores is True, the diagnosis will be a list of tuples,
          where the first element is the index and the second is the violation ratio.
        """
        if self.diagnosis is None:
            self.diagnosis = [(node_index, self.get_node_violation_difference(node_index)) for node_index in self.mapped_tree.tree_dict.keys()]
            self.diagnosis = sorted(self.diagnosis, key=lambda x: x[1], reverse=True)
        if retrive_scores:
            return self.diagnosis
        return [node_index for node_index, _ in self.diagnosis]