from APPETITE.MappedDecisionTree import MappedDecisionTree

from .AIndependentFixer import AIndependentFixer

class AllNodesFixer(AIndependentFixer):
    def fix_tree(self
     ) -> tuple[MappedDecisionTree, list[int]]:
        """
        Fix the decision tree.

        Returns:
            MappedDecisionTree: The fixed decision tree.
            list[int]: The indices of the faulty nodes.
        """
        # print(f"Fixing faulty nodes: {self.faulty_nodes}")
        for faulty_node_index in self.faulty_nodes:
            data_reached_faulty_node = self._filter_data_reached_fault(faulty_node_index)
            self.fix_faulty_node(faulty_node_index, data_reached_faulty_node)
        self.tree_already_fixed = True
        return super().fix_tree()