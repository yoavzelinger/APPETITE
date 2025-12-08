from sklearn.tree import DecisionTreeClassifier

from APPETITE.SubTreeReplaceableDecisionTree import SubTreeReplaceableDecisionTree

from .AFixer import AFixer

class SubTreeRetrainingFixer(AFixer):
    alias = "subtree_retrain"
    def fix_tree(self) -> tuple[DecisionTreeClassifier, list[int]]:
        """
        Fix the decision tree.

        Returns:
            DecisionTreeClassifier: The fixed decision tree.
            list[int]: The indices of the faulty nodes.
        """
        # print(f"Fixing faulty nodes: {self.faulty_nodes}")
        self.fixed_tree = SubTreeReplaceableDecisionTree(self.original_mapped_tree, self.faulty_nodes)
        self.fixed_tree.fit(self.X, self.y)
        
        return super().fix_tree()