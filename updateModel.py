from sklearn.tree import _tree, export_text
#from sklearn.tree.export import export_text
from SFL import PARENTS

def change_tree_threshold(tree, nodes, thresholds):
    for i in range(len(nodes)):
        tree.tree_.threshold[nodes[i]] = thresholds[i]
    return tree

def change_tree_selection(tree, nodes):
    for node in nodes:
        left_child = tree.tree_.children_left[node]
        right_child = tree.tree_.children_right[node]
        tree.tree_.children_left[node] = right_child
        tree.tree_.children_right[node] = left_child
        print("node id: {}".format(node))
        print("left child: {}".format(left_child))
        print("right child: {}".format(right_child))
    return tree

def find_nodes_threshold_from_diagnosis(model, diagnosis_features, features_diff):
    diagnosis_features = list(int(x / 2) for x in diagnosis_features)

    nodes = list()
    thresholds = list()

    node_count = model.tree_.node_count
    for i in range(node_count):
        feature = model.tree_.feature[i]
        if feature in diagnosis_features:
            nodes.append(i)
            new_threshold = model.tree_.threshold[i] + features_diff[feature]
            thresholds.append(new_threshold)
            # print("update node {} with feature {}, change threshold by {}".format(i, feature, new_threshold))

    return nodes, thresholds

def print_tree_rules(tree, feature_names):
    tree_rules = export_text(tree, feature_names=feature_names)
    print(tree_rules)

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print( "def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print( "{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print( "{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print( "{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)

def change_nodes_threshold(model, nodes, features_diff):
    for node in nodes:
        feature = model.tree_.feature[node]
        if feature == -2: # node is a leaf - no threshold
            continue
        # print(f"node {node} feature is: {feature}")
        new_threshold = model.tree_.threshold[node] + features_diff[feature]
        model.tree_.threshold[node] = new_threshold
    return model

def change_nodes_by_type(model, nodes,feature_types, features_diff):
    binary_categorical = list()
    numeric = list()
    for node in nodes:
        feature = model.tree_.feature[node]
        if feature == -2:  # node is a leaf
            continue
        f_type = feature_types[feature]
        if f_type == "numeric":
            numeric.append(node)
        elif f_type == "binary" or f_type == "categorical":
            binary_categorical.append(node)
    model = change_nodes_threshold(model, numeric, features_diff)
    model = change_tree_selection(model, binary_categorical)
    return model

def get_parents(nodes):
    parents = set()
    for node in nodes:
        parent = PARENTS[node]
        parents.add(parent)
    return parents
