from sklearn.tree import DecisionTreeClassifier

from APPETITE.MappedDecisionTree import MappedDecisionTree

from .AIndependentFixer import AIndependentFixer

class AllNodesFixer(AIndependentFixer):
    alias = "fix_all"

    def fix_tree(self) -> tuple[DecisionTreeClassifier, list[int]]:
        """
        Fix the decision tree.

        Returns:
            DecisionTreeClassifier: The fixed decision tree.
            list[int]: The indices of the faulty nodes.
        """
        # print(f"Fixing faulty nodes: {self.faulty_nodes}")
        for faulty_node_index in self.faulty_nodes:
            X_reached_faulty_node, _ = self._filter_data_reached_fault(faulty_node_index)
            self.fix_faulty_node(faulty_node_index, X_reached_faulty_node)
        return super().fix_tree()