from sklearn.tree import DecisionTreeClassifier

from APPETITE.SubTreeReplaceableDecisionTree import SubTreeReplaceableDecisionTree

from .AFixer import AFixer

class SubTreeRetrainingFixer(AFixer):
    def fix_tree(self) -> tuple[DecisionTreeClassifier, list[int]]:
        """
        Fix the decision tree.

        Returns:
            DecisionTreeClassifier: The fixed decision tree.
            list[int]: The indices of the faulty nodes.
        """
        # print(f"Fixing faulty nodes: {self.faulty_nodes}")
        fixed_decision_tree = SubTreeReplaceableDecisionTree(self.mapped_tree)
        
        for faulty_node_index in self.faulty_nodes:
            X_reached_faulty_node, y_reached_faulty_node = self._filter_data_reached_fault(faulty_node_index)
            fixed_decision_tree.replace_subtree(faulty_node_index, X_reached_faulty_node, y_reached_faulty_node)
        
        self.tree_already_fixed = True
        return fixed_decision_tree, self.faulty_nodes