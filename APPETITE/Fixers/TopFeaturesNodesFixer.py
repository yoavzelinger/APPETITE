from sklearn.tree import DecisionTreeClassifier

from APPETITE.MappedDecisionTree import MappedDecisionTree

from .AIndependentFixer import AIndependentFixer

class TopFeaturesNodesFixer(AIndependentFixer):
    alias = "per_features_top_tweak"
    def fix_tree(self) -> tuple[DecisionTreeClassifier, list[int]]:
        """
        Fix the decision tree.

        Returns:
            DecisionTreeClassifier: The fixed decision tree.
            list[int]: The indices of the faulty nodes.
        """
        top_features_faults: dict[str, tuple[int, int]] = {}
        for current_node in map(self.original_mapped_tree.get_node, self.faulty_nodes):
            _, current_feature_fault_top_depth = top_features_faults.get(current_node.feature, (None, float('inf')))
            if current_node.depth < current_feature_fault_top_depth:
                top_features_faults[current_node.feature] = (current_node.sk_index, current_node.depth)
        for faulty_node_index, _ in top_features_faults.values():
            X_reached_faulty_node, _ = self._filter_data_reached_fault(faulty_node_index)
            self.fix_faulty_node(faulty_node_index, X_reached_faulty_node)
        return super().fix_tree()