from sklearn.tree import DecisionTreeClassifier

from APPETITE.MappedDecisionTree import MappedDecisionTree

from .AIndependentFixer import AIndependentFixer

class TopNodeFixer(AIndependentFixer):
    def fix_tree(self) -> tuple[DecisionTreeClassifier, list[int]]:
        """
        Fix the decision tree.

        Returns:
            DecisionTreeClassifier: The fixed decision tree.
            list[int]: The indices of the faulty nodes.
        """
        top_faulty_node_index = min(self.faulty_nodes, key=lambda node_index: self.original_mapped_tree.get_node(node_index).depth)
        X_reached_top_faulty_node, _ = self._filter_data_reached_fault(top_faulty_node_index)
        self.fix_faulty_node(top_faulty_node_index, X_reached_top_faulty_node)
        self.tree_already_fixed = True
        return super().fix_tree()