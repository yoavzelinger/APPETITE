from barinel_utils import *
from .SFLDT import SFLDT

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

class BARINEL(SFLDT):
    def __init__(mapped_tree, X, y):
        super().__init__(mapped_tree, X, y)

    def get_diagnosis(self):