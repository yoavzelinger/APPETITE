"""
BARINEL implementation.
"""
from .DDIFMAS import _barinel_diagnosis, _rank_diagnoses as rank_diagnoses_v1
__author__ = "Avi Natan"
__homepage__ = "https://github.com/avi-natan/DDIFMAS"

def get_candidates(spectrum):
    """
    Get the candidates.

    Parameters:
    spectrum (ndarray): The spectrum.

    Returns:
        list: The candidates.
    """
    return _barinel_diagnosis(spectrum, [])


"""
New BARINEL implementation with fuzzy support.
"""
from APPETITE import Constants as constants
from .ranking_utils import rank_diagnoses as rank_diagnoses_v2

def rank_diagnoses(spectrum, diagnoses, components_prior_probabilities=None):
    """
    Rank the diagnoses.

    Parameters:
    diagnoses (ndarray): The diagnoses.
    spectrum (ndarray): The spectrum.
    fuzzy_error_vector (ndarray): The fuzzy error vector.

    Returns:
        ndarray: The ranked diagnoses.
    """
    if constants.BARINEL_RANKING_ALGORITHM == "V1":
        return rank_diagnoses_v1(spectrum, diagnoses, constants.GRADIENT_STEP)
    return rank_diagnoses_v2(spectrum, diagnoses, components_prior_probabilities)


__all__ = ["get_candidates", "rank_diagnoses"]