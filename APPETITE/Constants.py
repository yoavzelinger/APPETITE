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
one_hot_encoding = False

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

SFLDT_DEFAULT_SIMILARITY_MEASURES = "faith"
DEFAULT_FIXING_DIAGNOSER = ("SFLDT", "STAT_SFLDT", "SFLDT_Features")    # SFLDT Types
# DEFAULT_FIXING_DIAGNOSER = ("STAT_SFLDT", "BARINEL", "BARINEL_Paths_After", "STAT_BARINEL_Paths_After", "BARINEL_Paths_Difference", "STAT_BARINEL_Paths_Difference", "BARINEL_Features", "BARINEL_Features_Paths_After", "BARINEL_Features_Paths_Difference", "SFLDT_Features") # All relevant diagnosers

BARINEL_COMPONENT_PRIOR_PROBABILITY = 1 / 1000

# Choose the BARINEL ranking algorithm
# V1: discrete error ranking algorithm taken from DDIFMAS
# V2: new ranking algorithm, supporting fuzzy error and participation matrix
BARINEL_RANKING_ALGORITHM = "V2"

GRADIENT_STEP = 0.5

BARINEL_STAT_TYPE = "DIFFERENCE" # in ["BEFORE", "AFTER", "DIFFERENCE"]

DIAGNOSIS_ALGORITHM = "BARINEL" # in ["SFLDT", "BARINEL"]
USE_FUZZY_PARTICIPATION = False # Use fuzzy participation matrix
USE_FUZZY_ERROR = False # Use fuzzy error vector

BARINEL_STD_THRESHOLD = 0.5

SINGLE_DIAGNOSER_TYPE_NAME = "single_diagnoser"
MULTIPLE_DIAGNOSER_TYPE_NAME = "multiple_diagnoser"

DIAGNOSER_TYPES = (SINGLE_DIAGNOSER_TYPE_NAME, MULTIPLE_DIAGNOSER_TYPE_NAME)