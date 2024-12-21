from barinel_utils import *

def barinel(spectrum: list[list[int]]) -> list[list[int]]:
    """
    Perform the Barinel diagnosis algorithm on the given spectrum.

    Parameters:
    spectrum (list[list[int]]): The spectrum.

    Returns:
    list[list[int]]: All the diagnoses.
    """
    diagnoses, _ = _barinel_diagnosis(spectrum)
    return diagnoses