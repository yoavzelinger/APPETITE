from .STAT import STAT
from .SFLDT import SFLDT, SFLDT_DEFAULT_SIMILARITY_MEASURES
from .STAT_SFLDT import STAT_SFLDT
from .BARINEL import BARINEL, get_barinel_diagnoses
from .STAT_BARINEL import STAT_BARINEL
from .BARINEL_Combo import BARINEL_Combo

# The diagnosers dictionary - format: {diagnoser name: (diagnoser class, (diagnoser default parameters tuple))}
diagnosers_dict = {
    "SFLDT": (SFLDT, (SFLDT_DEFAULT_SIMILARITY_MEASURES, )),
    "STAT": (STAT, ()),
    "STAT_SFLDT": (STAT_SFLDT, (SFLDT_DEFAULT_SIMILARITY_MEASURES, )),
    "BARINEL": (BARINEL, ()),
    "STAT_BARINEL": (STAT_BARINEL, ()),
    "BARINEL_Combo": (BARINEL_Combo, ())
}

def get_diagnoser(diagnoser_name: str, 
                  *diagnoser_parameters: object
 ) -> tuple[object, tuple[object]]:
    """
    Get the diagnoser class and the diagnoser default parameters tuple.

    Parameters:
    diagnoser_name (str): The diagnoser name.
    diagnoser_parameters (tuple[object]): The diagnoser parameters.

    Returns:
    tuple[object, tuple[object]]: The diagnoser class and the diagnoser default parameters tuple.
    """
    assert diagnoser_name in diagnosers_dict, f"Diagnoser {diagnoser_name} is not supported"

    diagnoser_class, diagnoser_default_parameters = diagnosers_dict[diagnoser_name]
    if diagnoser_parameters is None:
        diagnoser_parameters = diagnoser_default_parameters
    return diagnoser_class, diagnoser_parameters

__all__ = ["STAT", "SFLDT", "SFLDT_DEFAULT_SIMILARITY_MEASURES", "STAT_SFLDT", "BARINEL", "get_barinel_diagnoses", "STAT_BARINEL", "BARINEL_Combo", "get_diagnoser"]