from os import path as os_path
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
    TESTING_DIAGNOSERS_CONFIGURATION_FILE_PATH = os_path.join(__name__, tester_constants.TESTING_DIAGNOSERS_CONFIGURATION_FILE_NAME)
    with open(f"{TESTING_DIAGNOSERS_CONFIGURATION_FILE_PATH}.json", "r") as testing_diagnosers_configuration_file:
        diagnosers_data = load_json(testing_diagnosers_configuration_file)
    return diagnosers_data