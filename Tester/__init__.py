from json import load as load_json
from .single_test import *
import Tester.TesterConstants as tester_constants

def load_testing_diagnosers_data( 
) -> list[dict[str, object]]:
    """
    Load the testing diagnosers from the JSON file.
    
    Returns:
        list[dict[str, object]]: The testing diagnosers.
    """
    with open(f"{tester_constants.TESTING_DIAGNOSERS_CONFIGURATION_FILE_NAME}.json", "r") as testing_diagnosers_configuration_file:
        diagnosers_data = load_json(testing_diagnosers_configuration_file)
    return diagnosers_data