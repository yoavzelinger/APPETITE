import itertools

import numpy as np

from DataSet import DataSet
from buildModel import build_model, map_tree, prune_tree, print_tree_rules


def predict_sample(sample, tree, active_nodes, f="binary"):
    node = 0
    values = predict_sample_from_node(node, sample, tree, active_nodes)
    if f == "binary":
        class_num = np.argmax(values)
        prediction = tree["classes"][class_num]
        return prediction
    elif f == "confident":
        class_num = np.argmax(values)
        n = values[class_num]
        confident = n / values.sum()
        return confident

def predict_sample_from_node(node, sample, tree, active_nodes):
    # if node is leaf - return classes
    if tree[node]["left"] == -1:  # leaf
        return tree[node]["value"]

    # if subtree doesn't contain active nodes - return classes
    subtree = set(tree[node]["subtree"])
    active_subtree_nodes = subtree.intersection(set(active_nodes))
    if len(active_subtree_nodes) == 0:
        return tree[node]["value"]

    # else - calculate recursively
    left_child = tree[node]["left"]
    right_child = tree[node]["right"]
    if node in active_nodes:
        # check condition - decide left or right
        feature = tree[node]["feature"]
        threshold = tree[node]["threshold"]
        if sample[feature] <= threshold:
            child = left_child
        else:  # sample[feature] > threshold
            child = right_child
        return predict_sample_from_node(child, sample, tree, active_nodes)

    else:  # node is not active - sum both child results
        c_right = predict_sample_from_node(right_child, sample, tree, active_nodes)
        c_left = predict_sample_from_node(left_child, sample, tree, active_nodes)
        res = c_right + c_left
        return res

def get_all_permutations(tree):
    node_list = list(tree_rep.keys())
    node_list.remove("classes")
    non_leaf_nodes = list(filter(lambda n: tree[n]["left"] != -1, node_list))
    n = len(non_leaf_nodes)
    permuts = list()
    for i in range(n+1):
        permuts += itertools.combinations(non_leaf_nodes, r=i)
    return permuts

def calculate_tree_values(tree):
    results = {}
    node_list = list(tree_rep.keys())
    node_list.remove("classes")
    leaf_nodes = list(filter(lambda n: tree[n]["left"] == -1, node_list))
    non_leaf_nodes = list(filter(lambda n: tree[n]["left"] != -1, node_list))
    non_leaf_nodes = sorted(non_leaf_nodes, reverse=True)

    for leaf in leaf_nodes:
        results[leaf] = {tuple(): tree[leaf]["value"]}

    for node in non_leaf_nodes:
        results[node] = {tuple(): tree[node]["value"]}
        subtree = set(tree[node]["subtree"])
        subtree = subtree.intersection(set(non_leaf_nodes))

        n = len(subtree)
        permuts = list()
        for i in range(n + 1):
            permuts += itertools.combinations(subtree, r=i)

        for p in permuts:
            results[node][p] = {}
            options = list(itertools.product(["L", "R"], repeat=len(p)))

            # go trough every Left Right option
            for opt in options:
                mode = dict(zip(p,opt))
                # if node is active calculate left or right
                if node in p:
                    if mode[node] == "L":
                        child = tree[node]["left"]
                    else:  # R
                        child = tree[node]["right"]
                    active_subtree = tuple(set(tree[child]["subtree"]).intersection(set(p)))
                    child_mode = tuple([mode[i] for i in active_subtree])
                    ans = results[child][active_subtree][child_mode]

                # if node is inactive - sum left right
                else:
                    left_child = tree[node]["left"]
                    active_subtree_l = tuple(set(tree[left_child]["subtree"]).intersection(set(p)))
                    child_mode_l = tuple([mode[i] for i in active_subtree_l])
                    ans_l = results[left_child][active_subtree_l][child_mode_l]

                    right_child = tree[node]["right"]
                    active_subtree_r = tuple(set(tree[right_child]["subtree"]).intersection(set(p)))
                    child_mode_r = tuple([mode[i] for i in active_subtree_r])
                    ans_r = results[right_child][active_subtree_r][child_mode_r]

                    ans = ans_l+ans_r

                results[node][p][opt] = ans

    return results

if __name__ == '__main__':
    dataset = DataSet("data/Classification_Datasets/breast-cancer-wisc-diag.csv", "diagnosis_check", None, None,
                       (0.7, 0.1, 0.2), name="breast-cancer-wisc-diag.csv", to_shuffle=True)

    # build tree
    concept_size = dataset.before_size
    target = dataset.target
    feature_types = dataset.feature_types
    train = dataset.data.iloc[0:int(0.9 * concept_size)].copy()
    validation = dataset.data.iloc[int(0.9 * concept_size):concept_size].copy()
    model = build_model(train, dataset.features, dataset.target, val_data=validation)
    tree_rep = map_tree(model)
    model = prune_tree(model, tree_rep)
    print("TREE:")
    print_tree_rules(model, dataset.features)
    tree_rep = map_tree(model)

    all_ans = calculate_tree_values(tree_rep)
    print(all_ans)

    permutes = get_all_permutations(tree_rep)
    results = {}
    sample = dataset.data.loc[400]
    for p in permutes:
        res = predict_sample(sample, tree_rep, p)
        results[p] = res

    print(results)





