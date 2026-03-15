import numpy as np
import pandas as pd

from math import factorial
from itertools import combinations

import APPETITE.Constants as constants
from APPETITE.ModelMapping.ATreeBasedMappedModel import ATreeBasedMappedModel
from APPETITE.ModelMapping.TreeNodeComponent import TreeNodeComponent

class NodeSHAPFunctions:
    @staticmethod
    def confidence(class_distribution_with, class_distribution_without):
        def _confidence(class_distribution):
            total = class_distribution.sum()
            if total == 0:
                return 0.0
            return class_distribution.max() / total
        return _confidence(class_distribution_with) - _confidence(class_distribution_without)
    
    @staticmethod
    def gini(class_distribution_with, class_distribution_without):
        def _gini(class_distribution):
            total = class_distribution.sum()
            if total == 0:
                return 0.0
            probabilities = class_distribution / total
            return 1.0 - np.sum(probabilities ** 2)
        return _gini(class_distribution_without) - _gini(class_distribution_with)
    
    @staticmethod
    def entropy(class_distribution_with, class_distribution_without):
        def _entropy(class_distribution):
            total = class_distribution.sum()
            if total == 0:
                return 0.0
            probabilities = class_distribution / total
            probabilities = probabilities[probabilities > 0]
            return -np.sum(probabilities * np.log2(probabilities))
        return _entropy(class_distribution_without) - _entropy(class_distribution_with)
    
    @staticmethod
    def prediction(class_distribution_with, class_distribution_without):
        return float(np.argmax(class_distribution_with) != np.argmax(class_distribution_without))
    
    functions_map = {
        constants.NodeSHAPFunctionType.Confidence: confidence,
        constants.NodeSHAPFunctionType.Gini: gini,
        constants.NodeSHAPFunctionType.Entropy: entropy,
        constants.NodeSHAPFunctionType.Prediction: prediction,
    }

    def __class_getitem__(_, node_shap_function_type: str | constants.NodeSHAPFunctionType):
        if isinstance(node_shap_function_type, str):
            node_shap_function_type = constants.NodeSHAPFunctionType[node_shap_function_type]
        assert node_shap_function_type in NodeSHAPFunctions.functions_map, f"Unsupported Node SHAP function type: {node_shap_function_type}"
        return NodeSHAPFunctions.functions_map[node_shap_function_type]

def _get_node_shap_function(criterion: str):
    """
    Get the Node-SHAP function based on the criterion of the decision tree.

    Parameters:
        criterion (str): The criterion used in the decision tree (e.g., "gini", "entropy").
    
    Returns:
        function: The corresponding Node-SHAP function.
    """
    current_function_type = constants.DEFAULT_NODE_SHAP_FUNCTION_TYPE
    if current_function_type == constants.NodeSHAPFunctionType.Criterion:
        current_function_type = criterion
    return NodeSHAPFunctions[current_function_type]

def _get_class_distribution(mapped_model: ATreeBasedMappedModel, node: TreeNodeComponent, num_classes: int) -> np.ndarray:
    """Get the class distribution for a node from the sklearn tree's value array."""
    return mapped_model.model.tree_.value[node.component_index].flatten()[:num_classes]

def _sorted_nodes(nodes) -> list[TreeNodeComponent]:
    """Sort nodes by component_index for deterministic iteration."""
    return sorted(nodes, key=lambda n: n.component_index)

def compute_tree_analysis(mapped_model: ATreeBasedMappedModel) -> dict:
    """
    Build the Tree Analysis table using bottom-up dynamic programming.

    For each node, computes class distributions for all combinations of
    (active_node_subset, left-right_allocation).

    A subset is represented as a frozenset of TreeNodeComponent objects (internal nodes).
    A left-right allocation is represented as a frozenset of TreeNodeComponent objects
    that the sample goes RIGHT on within the subset.

    Parameters:
        mapped_model: The mapped tree based model.

    Returns:
        dict: TA[root] mapping (subset, right_set) → class_distribution (numpy array).
    """
    num_classes = mapped_model.model.tree_.n_classes[0]

    # TA per node: dict mapping (frozenset_subset, frozenset_right_set) → np.array
    tree_analysis: dict[TreeNodeComponent, dict] = {}

    def build_node_analysis(node: TreeNodeComponent):
        class_distribution = _get_class_distribution(mapped_model, node, num_classes)
        node_analysis = {}
        empty = frozenset()

        # Base: empty subset → full class distribution of this subtree root
        node_analysis[(empty, empty)] = class_distribution.copy()

        if node.is_terminal():
            tree_analysis[node] = node_analysis
            return

        # Recursively build children first
        build_node_analysis(node.left_child)
        build_node_analysis(node.right_child)

        left_child_analysis = tree_analysis[node.left_child]
        right_child_analysis = tree_analysis[node.right_child]
        left_internals = frozenset(node.left_child.get_all_internals())
        right_internals = frozenset(node.right_child.get_all_internals())
        node_internals = frozenset(node.get_all_internals())

        # Iterate over all subsets of internal nodes in this subtree
        node_internals_list = _sorted_nodes(node_internals)
        for subset_size in range(1, len(node_internals_list) + 1):
            for subset_tuple in combinations(node_internals_list, subset_size):
                subset = frozenset(subset_tuple)
                # Determine which nodes in subset belong to left/right subtrees
                left_subset = subset & left_internals
                right_subset = subset & right_internals
                node_in_subset = node in subset

                # Generate all left-right allocations for nodes in subset
                subset_list = _sorted_nodes(subset)
                for right_mask in range(1 << len(subset_list)):
                    right_set = frozenset(
                        subset_list[bit] for bit in range(len(subset_list)) if (right_mask >> bit) & 1
                    )

                    if node_in_subset:
                        # Node is active: sample follows constraint direction
                        if node in right_set:
                            # Goes right
                            child_subset = right_subset
                            child_right_set = right_set & right_internals
                            result = right_child_analysis.get((child_subset, child_right_set))
                        else:
                            # Goes left
                            child_subset = left_subset
                            child_right_set = right_set & left_internals
                            result = left_child_analysis.get((child_subset, child_right_set))
                    else:
                        # Node not active: goes both ways
                        left_right_set = right_set & left_internals
                        right_right_set = right_set & right_internals
                        left_result = left_child_analysis.get((left_subset, left_right_set))
                        right_result = right_child_analysis.get((right_subset, right_right_set))
                        if left_result is not None and right_result is not None:
                            result = left_result + right_result
                        else:
                            result = None

                    if result is not None:
                        node_analysis[(subset, right_set)] = result.copy()

        tree_analysis[node] = node_analysis

    build_node_analysis(mapped_model.root)
    return tree_analysis[mapped_model.root]


def compute_node_shap_values(
    mapped_model: ATreeBasedMappedModel,
    X: pd.DataFrame,
    inverse_spectra_map: dict[TreeNodeComponent, int],
) -> np.ndarray:
    """
    Compute Node-SHAP values for all samples and all nodes.

    Parameters:
        mapped_model: The mapped tree based model.
        X: The data samples.
        inverse_spectra_map: Mapping from TreeNodeComponent to spectra index,
            used to place results in the correct row of the output matrix.

    Returns:
        np.ndarray: Shape (num_components, num_samples). node_shap_values[spectra_idx, sample_idx].
    """
    tree_analysis = compute_tree_analysis(mapped_model)
    internal_nodes = set(filter(TreeNodeComponent.is_internal, mapped_model))
    samples_count = len(X)

    current_node_shap_function_type = constants.DEFAULT_NODE_SHAP_FUNCTION_TYPE
    if current_node_shap_function_type == constants.NodeSHAPFunctionType.Criterion:
        current_node_shap_function_type = mapped_model.model.criterion    
    current_shap_function = NodeSHAPFunctions[current_node_shap_function_type]

    # Pre-compute sample left-right allocations: for each sample, which internal nodes go right
    sample_right_sets = []
    for sample_index in range(samples_count):
        right_nodes = set()
        for node in internal_nodes:
            if X[node.feature].iloc[sample_index] > node.threshold:
                right_nodes.add(node)
        sample_right_sets.append(frozenset(right_nodes))

    shap_values = np.zeros((len(mapped_model), samples_count))

    get_weight = lambda subset_size: factorial(subset_size) * factorial(len(internal_nodes) - subset_size - 1) / factorial(len(internal_nodes))

    for node in internal_nodes:
        for sample_right in sample_right_sets:
            
            # Iterate over all subsets of other nodes
            for subset_size in range(len(internal_nodes)):
                for subset_tuple in combinations(internal_nodes - {node}, subset_size):
                    subset = frozenset(subset_tuple)
                    subset_with_node = subset | {node}

                    # left-right allocation for S (restricted to sample's directions)
                    right_set_without = sample_right & subset
                    right_set_with = sample_right & subset_with_node

                    class_distribution_without = tree_analysis.get((subset, right_set_without))
                    class_distribution_with = tree_analysis.get((subset_with_node, right_set_with))

                    if class_distribution_without is not None and class_distribution_with is not None:
                        shap_values[inverse_spectra_map[node], sample_index] += get_weight(subset_size) * current_shap_function(class_distribution_with, class_distribution_without)

    return shap_values
