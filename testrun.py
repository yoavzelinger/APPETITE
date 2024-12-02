from DataManagementTools.Dataset import Dataset
from DataManagementTools.DriftSimulation import single_feature_concept_drift_generator, multiple_features_concept_drift_generator
from DecisionTreeTools.MappedDecisionTree import MappedDecisionTree
from DecisionTreeTools import DecisionTreeClassifierBuilder

DIRECTORY = "data\\real\\"
FILE_NAME = "iris.data"

def get_dataset():
    return Dataset(DIRECTORY + FILE_NAME)

def get_example_tree():
    dataset = get_dataset()
    features = list(dataset.feature_types.keys())
    target = dataset.target
    return DecisionTreeClassifierBuilder.build(dataset.data, features, target)

def get_mapped_tree():
    return MappedDecisionTree(get_example_tree())

def get_partition_drift_generator():
    dataset = get_dataset()
    features = list(dataset.feature_types.keys())
    drift_features = features[0: 2]
    yield from dataset.partition_drift_generator(drift_features)

get_dataset()
get_example_tree()
get_mapped_tree()
get_partition_drift_generator()