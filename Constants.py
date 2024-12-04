
# Dataset partitions sizes
CONCEPT_PROPORTION = 0.7
DRIFT_PROPOTION = 0.1
TEST_PROPORTION = 0.2
PROPORTIONS_TUPLE = (CONCEPT_PROPORTION, DRIFT_PROPOTION, TEST_PROPORTION)

VALIDATION_SIZE = 0.2

# Random state
RANDOM_STATE = 7

# Grid search parameters
CROSS_VALIDATION_SPLIT_COUNT = 5
_CRITERIONS = ["gini", "entropy"]
_MAX_LEAF_NODES = [10, 20, 30]
PARAM_GRID = {
    "criterion": _CRITERIONS,
    "max_leaf_nodes": _MAX_LEAF_NODES
}

# Drift severity levels
NUMERIC_DRIFT_SEVERITIES = (
    -2, # Severity 3
    -1, # Severity 2
    -0.5, # Severity 1
    0.5, # Severity 1
    1, # Severity 2
    2 # Severity 3
)
CATEGORICAL_DRIFT_SEVERITIES = (
    0.3, # Severity 1
    0.5, 0.7, # Severity 2
    0.9 # Severity 3
)