from sys import float_info
EPSILON = float_info.epsilon

# Random state
RANDOM_STATE = 7

BARINEL_COMPONENT_PRIOR_PROBABILITY = 1 / 1000

# BARINEL ranking v1
BARINEL_GRADIENT_STEP = 0.5

# Choose the BARINEL ranking algorithm
# V1: discrete error ranking algorithm taken from DDIFMAS
# V2: new ranking algorithm, supporting fuzzy error and participation matrix, and custom prior probabilities
BARINEL_RANKING_ALGORITHM_VERSION = 2

GRADIENT_STEP = 0.5

DEFAULT_COMBINE_STAT = True # Combine STAT Diagnoses
DEFAULT_USE_SHAP_CONTRIBUTION = False # Try to use SHAP contributions for the participation
DEFAULT_AGGREGATE_TESTS_BY_PATHS = False # Aggregate tests by paths in the decision tree
DEFAULT_GROUP_FEATURE_NODES = False # Group feature nodes to a single component
DEFAULT_COMBINE_PRIOR_CONFIDENCE = False # Use the prior confidence in Error vector calculation

BARINEL_THRESHOLD_ABOVE_STD_RATE = 0.5 # setting the error threshold (from which tests considered as failed) for mean + std * BARINEL_THRESHOLD_ABOVE_STD_RATE