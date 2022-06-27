import pandas as pd

from DataSet import DataSet
from buildModel import build_model
from updateModel import print_tree_rules

source_path = "data/real/winequality-white.csv"
data = pd.read_csv(source_path)
print(list(data.columns[:-1]))
print(type(data.columns[-1]))

target = data.columns[-1]
features = list(data.columns[:-1])
dataset = DataSet(source_path, "diagnosis_check", target, 2000, ["numeric"]*len(features))

model = build_model(dataset.data.iloc[0:2000], dataset.features, dataset.target, to_split=False)

print("TREE:")
print_tree_rules(model, dataset.features)


def map_tree(tree):
    tree_representation = {0: {"depth": 0,
                "parent": -1}}
    nodes_to_check = [0]
    while len(nodes_to_check) > 0:
        node = nodes_to_check.pop(0)

        left_child = tree.tree_.children_left[node]
        tree_representation[node]["left"] = left_child
        if left_child != -1:
            tree_representation[left_child] = {"parent":node,
                                               "type": "left"}
            nodes_to_check.append(left_child)
        right_child = tree.tree_.children_right[node]
        tree_representation[node]["right"] = right_child
        if right_child != -1:
            tree_representation[right_child] = {"parent":node,
                                               "type": "right"}
            nodes_to_check.append(right_child)

        tree_representation[node]["feature"] = model.tree_.feature[node]
        tree_representation[node]["threshold"] = model.tree_.threshold[node]

        if node != 0:
            parent = tree_representation[node]["parent"]
            tree_representation[node]["depth"] = tree_representation[parent]["depth"] + 1
            parent_cond = tree_representation[parent]["condition"]
            sign = "<=" if tree_representation[node]["type"] == "left" else ">"
            cond = f"{model.tree_.feature[parent]} {sign} {model.tree_.threshold[parent]}"
            tree_representation[node]["condition"] = parent_cond + [cond]
        else: #root
            tree_representation[node]["condition"] = []

    return tree_representation


node_list = list(range(model.tree_.node_count))
tree_rep = map_tree(model)
non_leaf_nodes = list(filter(lambda n: tree_rep[n]["left"] != -1, node_list))
print(non_leaf_nodes)
leaf_nodes = list(filter(lambda n: tree_rep[n]["left"] == -1, node_list))
print(leaf_nodes)

def manipulate_node(node, dataset):
    feature_to_change = tree_rep[node]["feature"]
    feature_to_change = dataset.features[int(feature_to_change)]
    conditions = tree_rep[node]["condition"]
    verification_data = dataset.data.iloc[0:2000]
    filtered_data = dataset.data.iloc[2000:]

    # filtering only node data
    for cond in conditions:
        feature, sign, thresh = cond.split()
        thresh = float(thresh)
        feature_name = dataset.features[int(feature)]
        if sign == ">":
            filtered_data = filtered_data[filtered_data[feature_name] > thresh]
            verification_data = verification_data[verification_data[feature_name] > thresh]
        else:
            filtered_data = filtered_data[filtered_data[feature_name] <= thresh]
            verification_data = verification_data[verification_data[feature_name] <= thresh]
    assert len(verification_data) == model.tree_.n_node_samples[node], f"bad condition - node: {node}, cond: {conditions}"

    # calculating statistics
    all_data = dataset.data
    mean = float(all_data.mean()[feature_to_change])
    std = float(all_data.std()[feature_to_change])

    # creating changes
    half_std_up = filtered_data[feature_to_change] + 0.5*std
    half_std_down = filtered_data[feature_to_change] - 0.5 * std
    one_std_up = filtered_data[feature_to_change] + 1 * std
    one_std_down = filtered_data[feature_to_change] - 1 * std
    feature_changes = [half_std_up, half_std_down, one_std_up, one_std_down]
    feature_changes_names = ["half_std_up", "half_std_down", "one_std_up", "one_std_down"]

    #saving changes to csv
    for i in range(len(feature_changes)):
        change = feature_changes[i]
        change_name = feature_changes_names[i]
        to_save = filtered_data
        to_save[feature_to_change] = change
        #change_name = f'{change=}'.split('=')[0]
        file_name = f'{dataset.name.split(".")[0]}_node_{node}_depth_{tree_rep[node]["depth"]}_{change_name}.csv'
        to_save.to_csv(file_name)

    #return filtered_data

print(model.tree_.n_node_samples)
for node in non_leaf_nodes:
    print(f"node: {node}, depth: {tree_rep[node]['depth']}")
    manipulate_node(node, dataset)
