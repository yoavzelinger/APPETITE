from sklearn.tree import DecisionTreeClassifier
import numpy as np

class MappedDecisionTree:
    class DecisionTreeNode:
        def __init__(self, 
                     index: int = 0,
                     parent = None, 
                     left_child = None, 
                     right_child = None,
                     feature: str | None = None,
                     threshold: float | None = None,
                     class_name: str | None = None
        ):
            self.index = index
            self.parent = parent
            self.update_children(left_child, right_child)
            self.feature = feature
            self.threshold = threshold
            self.class_name = class_name
            self.conditions_path = []
            self.depth = 0 if parent is None else parent.depth + 1

        def update_children(self, 
                            left_child: 'MappedDecisionTree.DecisionTreeNode', 
                            right_child: 'MappedDecisionTree.DecisionTreeNode'
        ) -> None:
            # A node can be either a terminal node (two children) or a non-terminal node (a leaf - no children)
            assert (left_child is None) == (right_child is None)
            self.left_child = left_child
            self.right_child = right_child

        def update_condition(self) -> None:
            if self.parent is None:
                return
            self.conditions_path = self.parent.conditions_path
            current_condition = {
                "feature": self.parent.feature,
                "sign": "<=" if self.is_left_child() else ">",
                "threshold": self.parent.threshold
            }
            self.conditions_path += [current_condition]

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
        
        def get_sibling(self):
            if self.parent is None:
                return None
            return self.parent.left_child if self.is_right_child() else self.parent.right_child
        
        def __repr__(self):
            return str(self.index)

    def __init__(self, 
                 sklearn_tree: DecisionTreeClassifier
    ):
        assert sklearn_tree is not None
        self.sklearn_tree = sklearn_tree
        self.node_count = sklearn_tree.tree_.node_count
        self.max_depth = sklearn_tree.tree_.max_depth
        self.n_classes = sklearn_tree.n_classes_

        self.tree_dict = self.map_tree()
        self.root = self.tree_dict[0]

    def map_tree(self, 
    ) -> dict[int, 'MappedDecisionTree.DecisionTreeNode']:
        sk_features = self.sklearn_tree.tree_.feature
        sk_thresholds = self.sklearn_tree.tree_.threshold
        sk_children_left = self.sklearn_tree.tree_.children_left
        sk_children_right = self.sklearn_tree.tree_.children_right
        sk_values = self.sklearn_tree.tree_.value
        sk_class_names = self.sklearn_tree.classes_
        
        tree_representation = {}
        nodes_to_check = [self.DecisionTreeNode(index=0)]

        while len(nodes_to_check):
            current_node = nodes_to_check.pop(0)
            current_node.update_condition()
            current_index = current_node.index
            tree_representation[current_index] = current_node
            left_child_index = sk_children_left[current_index]
            right_child_index = sk_children_right[current_node]

            if left_child_index == right_child_index:  # Leaf
                current_node_value = sk_values[current_index]
                class_name = np.argmax(current_node_value)
                class_name = sk_class_names[class_name]
                current_node.class_name = class_name
                continue

            current_node.feature, current_node.threshold = sk_features[current_index], sk_thresholds[current_index]
            right_child_index = sk_children_right[current_index]
            for child_index in (left_child_index, right_child_index):
                child_node = self.DecisionTreeNode(index=child_index, parent=current_node)
                nodes_to_check.append(child_node)
            current_node.update_children(*(nodes_to_check[-2: ]))

        return tree_representation