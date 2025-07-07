# Dataset partitions sizes
BEFORE_PROPORTION = 0.7
AFTER_PROPORTION = 0.1
TEST_PROPORTION = 0.2
PROPORTIONS_TUPLE = (BEFORE_PROPORTION, AFTER_PROPORTION, TEST_PROPORTION)
AFTER_WINDOW_SIZE = 1

VALIDATION_SIZE = 0.2

# Random state
RANDOM_STATE = 7

# one hot encode categorical features to modify decision trees creation
one_hot_encoding = True

# Grid search parameters
CROSS_VALIDATION_SPLIT_COUNT = 5
_CRITERIONS = ["gini", "entropy"]
_MAX_LEAF_NODES = [10, 20, 30]
PARAM_GRID = {
    "criterion": _CRITERIONS,
    "max_leaf_nodes": _MAX_LEAF_NODES
}

# Drift severity levels
NUMERIC_DRIFT_SEVERITIES = {
    1: (-0.5, 0.5),
    2: (-1, 1),
    3: (-2, 2)
}
CATEGORICAL_DRIFT_SEVERITIES = {
    1: (0.3, ),
    2: (0.5, 0.7),
    3: (0.9, )
}
DEFAULT_GENERATED_SEVERITY_LEVELS = (2, )

BARINEL_COMPONENT_PRIOR_PROBABILITY = 1 / 1000
# Choose the BARINEL ranking algorithm
# V1: discrete error ranking algorithm taken from DDIFMAS
# V2: new ranking algorithm, supporting fuzzy error and participation matrix
BARINEL_RANKING_ALGORITHM = "V2"

GRADIENT_STEP = 0.5

DEFAULT_COMBINE_STAT = True # Combine STAT Diagnoses
DEFAULT_FUZZY_PARTICIPATION = False # Use fuzzy participation matrix
DEFAULT_GROUP_TESTS_BY_PATHS = False # Group tests by paths in the decision tree
DEFAULT_FEATURE_COMPONENTS = False # Use features components
DEFAULT_USE_TESTS_CONFIDENCE = True # Use confidence in Error vector calculation
DEFAULT_MERGE_SINGULAR_DIAGNOSES = False # Merge singular diagnoses based on the features

BARINEL_THRESHOLD_ABOVE_STD_RATE = 0.5 # setting the error threshold (from which tests considered as failed) for mean + std * BARINEL_THRESHOLD_ABOVE_STD_RATE