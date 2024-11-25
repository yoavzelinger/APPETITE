from sklearn.tree import DecisionTreeClassifier
import numpy as np

class MappedDecisionTree:
    class DecisionTreeNode:
        def __init__(self, 
                     index: int = 0,
                     parent: 'MappedDecisionTree.DecisionTreeNode' | None = None, 
                     left_child: 'MappedDecisionTree.DecisionTreeNode' | None = None, 
                     right_child: 'MappedDecisionTree.DecisionTreeNode' | None = None,
                     feature: str | None = None,
                     threshold: int | None = None, # TODO - Verify Type
                     class_name: str | None = None
        ):
            self.index = index
            self.parent = parent
            self.update_children(left_child, right_child)
            self.feature = feature
            self.threshold = threshold
            self.class_name = class_name
            self.condition = []
            self.depth = 0 if parent is None else parent.depth + 1
            self.update_condition()

        def update_children(self, 
                            left_child: 'MappedDecisionTree.DecisionTreeNode', 
                            right_child: 'MappedDecisionTree.DecisionTreeNode'
        ) -> None:
            # A node can be either a terminal node (two children) or a non-terminal node (a leaf - no children)
            assert (left_child is None) == (right_child is None)
            self.left_child = left_child
            self.right_child = right_child

        def update_condition(self) -> None:
            if self.parent in None:
                return
            self.condition += self.parent.condition
            current_condition = {
                "feature": self.parent.feature,
                "threshold": self.parent.threshold,
                "sign": "<=" if self.is_left_child() else ">"
            }
            self.condition += [current_condition]

        def is_terminal(self) -> bool:
            return self.left_child is None
        
        def is_left_child(self) -> bool:
            if self.parent is None:
                return False
            return self.parent.left_child == self
        
        def is_right_child(self) -> bool:
            if self.parent is None:
                return False
            return not self.is_left_child()

    def __init__(self, 
                 sklearn_tree: DecisionTreeClassifier
    ):
        assert sklearn_tree is not None
        self.node_count = sklearn_tree.tree_.node_count
        self.max_depth = sklearn_tree.tree_.max_depth
        self.n_classes = sklearn_tree.n_classes_

        self.tree_dict = self.map_tree(sklearn_tree)
        self.root = self.tree_dict[0]

    def map_tree(self, 
                 sklearn_tree: DecisionTreeClassifier
    ) -> dict[int, 'MappedDecisionTree.DecisionTreeNode']:
        sk_features = sklearn_tree.tree_.feature
        sk_thresholds = sklearn_tree.tree_.threshold
        sk_children_left = sklearn_tree.tree_.children_left
        sk_children_right = sklearn_tree.tree_.children_right
        sk_values = sklearn_tree.tree_.value
        sk_class_names = sklearn_tree.classes_
        
        tree_representation = {}
        nodes_to_check = [self.DecisionTreeNode(index=0)]

        while len(nodes_to_check):
            current_node = nodes_to_check.pop(0)
            current_index = current_node.index
            tree_representation[current_index] = current_node
            left_child_index = sk_children_left[current_index]

            if left_child_index == -1:  # Leaf
                current_node_value = sk_values[current_index]
                class_name = np.argmax(current_node_value)
                class_name = sk_class_names[class_name]
                current_node.class_name = class_name
                continue

            current_node.feature, current_node.threshold = sk_features[current_index], sk_thresholds[current_index]
            right_child_index = sk_children_right[current_node]
            for child_index in (left_child_index, right_child_index):
                child_node = self.DecisionTreeNode(index=child_index, parent=current_node)
                nodes_to_check.append(child_node)
            current_node.update_children(*(nodes_to_check[-2: ]))

        return tree_representation