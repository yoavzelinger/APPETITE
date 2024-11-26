from sklearn.tree import DecisionTreeClassifier

class MappedDecisionTree:
    class DecisionTreeNode:
        def __init__(self, 
                     index: int = 0,
                     parent: 'MappedDecisionTree.DecisionTreeNode' | None = None, 
                     left_child: 'MappedDecisionTree.DecisionTreeNode' | None = None, 
                     right_child: 'MappedDecisionTree.DecisionTreeNode' | None = None,
                     feature: str | None = None,
                     threshold: int | None = None, # TODO - Verify Type
                     label: str | None = None
        ):
            self.index = index
            self.parent = parent
            self.update_children(left_child, right_child)
            if self.is_terminal():
                assert label is not None
            self.label = label
            self.feature = feature
            self.threshold = threshold
            self.condition = []
            self._calculate_condition()
            self.label = label
            self.depth = 0 if parent is None else parent.depth + 1

        def _calculate_condition(self):
            if self.parent in None:
                return
            self.condition += self.parent.condition
            current_condition = {
                "feature": self.parent.feature,
                "threshold": self.parent.threshold,
                "sign": "<=" if self.is_left_child() else ">"
            }
            self.condition += [current_condition]

            

        def update_children(self, 
                            left_child: 'MappedDecisionTree.DecisionTreeNode', 
                            right_child: 'MappedDecisionTree.DecisionTreeNode'
        ) -> None:
            # A node can be either a terminal node (two children) or a non-terminal node (a leaf - no children)
            assert (left_child is None) == (right_child is None)
            self.left_child = left_child
            self.right_child = right_child

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

    def map_tree(self, 
                 sklearn_tree: DecisionTreeClassifier
    ) -> dict[int, 'MappedDecisionTree.DecisionTreeNode']:
        threshold = sklearn_tree.tree_.threshold
        feature = sklearn_tree.tree_.feature
        children_left = sklearn_tree.tree_.children_left
        children_right = sklearn_tree.tree_.children_right
        value = sklearn_tree.tree_.value
        class_names = sklearn_tree.classes_
        
        tree_representation = {}
        nodes_to_check = [0]
        while len(nodes_to_check):
            # TODO: Continue
            node = nodes_to_check.pop(0)

            left_child_index = children_left[node]
            tree_representation[node]["left"] = left_child
            if left_child != -1:
                tree_representation[left_child] = {"parent": node,
                                                   "type": "left"}
                nodes_to_check.append(left_child)
            right_child = self.children_right[node]
            tree_representation[node]["right"] = right_child
            if right_child != -1:
                tree_representation[right_child] = {"parent": node,
                                                    "type": "right"}
                nodes_to_check.append(right_child)

            tree_representation[node]["feature"] = self.feature[node]
            tree_representation[node]["threshold"] = self.threshold[node]

            if node != 0:
                parent = tree_representation[node]["parent"]
                tree_representation[node]["depth"] = tree_representation[parent]["depth"] + 1
                parent_cond = tree_representation[parent]["condition"]
                sign = "<=" if tree_representation[node]["type"] == "left" else ">"
                cond = {
                    "feature": self.feature[parent],
                    "sign": sign,
                    "thresh": self.threshold[parent]
                }
                tree_representation[node]["condition"] = parent_cond + [cond]
            else:  # root
                tree_representation[node]["condition"] = []

            if left_child == -1:  # leaf
                value = self.value[node]
                class_name = np.argmax(value)
                class_name = class_names[class_name]
                tree_representation[node]["class"] = class_name

        return tree_representation