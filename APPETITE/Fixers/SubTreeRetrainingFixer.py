from sklearn.tree import DecisionTreeClassifier

import APPETITE.Constants as constants
from APPETITE.SubTreeReplaceableDecisionTree import SubTreeReplaceableDecisionTree

from .AFixer import AFixer

class SubTreeRetrainingFixer(AFixer):
    alias = "subtree_retrain"

    def __init__(self,
                 *args,
                 dependency_handling_type: str = constants.DEFAULT_SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPE,
                 use_prior_knowledge: bool = constants.DEFAULT_USE_OF_PRIOR_KNOWLEDGE,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dependency_handling_type: constants.SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPES = \
            constants.SUBTREE_RETRAINING_DEPENDENCY_HANDLING_TYPES[dependency_handling_type]
        
        self.use_prior_knowledge = use_prior_knowledge

    def fix_tree(self) -> tuple[DecisionTreeClassifier, list[int]]:
        """
        Fix the decision tree.

        Returns:
            DecisionTreeClassifier: The fixed decision tree.
            list[int]: The indices of the faulty nodes.
        """
        # print(f"Fixing faulty nodes: {self.faulty_nodes}")
        self.fixed_tree = SubTreeReplaceableDecisionTree(self.original_mapped_tree, self.faulty_nodes, dependency_handling_type=self.dependency_handling_type)
        
        if self.use_prior_knowledge:
            self.fixed_tree.fit(self.X, self.y, self.X_prior, self.y_prior)
        else:
            self.fixed_tree.fit(self.X, self.y)
        
        return super().fix_tree()