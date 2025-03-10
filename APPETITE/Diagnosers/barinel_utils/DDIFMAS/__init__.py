"""
BARINEL implementation.

Taken from DDIFMAS project.

Author: Avi Natan
Homepage: https://github.com/avi-natan/DDIFMAS
"""
from .methods_for_diagnosis import diagnosis_0 as _barinel_diagnosis
from .methods_for_ranking import ranking_0 as _rank_diagnoses

__author__ = "Avi Natan"
__homepage__ = "https://github.com/avi-natan/DDIFMAS"

__all__ = ["_barinel_diagnosis", "_rank_diagnoses"]