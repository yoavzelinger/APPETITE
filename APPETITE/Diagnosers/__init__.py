from .ADiagnoser import ADiagnoser
from .STAT import STAT
from .SFLDT import SFLDT, SFLDT_DEFAULT_SIMILARITY_MEASURES
from .STAT_SFLDT import STAT_SFLDT
from .BARINEL import BARINEL, get_barinel_diagnoses
from .STAT_BARINEL import STAT_BARINEL
from .BARINEL_Paths import BARINEL_Paths
from .BARINEL_Paths_After import BARINEL_Paths_After
from .BARINEL_Paths_Difference import BARINEL_Paths_Difference
from .STAT_BARINEL_Paths_After import STAT_BARINEL_Paths_After
from .STAT_BARINEL_Paths_Difference import STAT_BARINEL_Paths_Difference
from .BARINEL_Features import BARINEL_Features

# The diagnosers dictionary - format: {diagnoser name: (diagnoser class, (diagnoser default parameters tuple))}
diagnosers_dict = {
    "SFLDT": (SFLDT, (SFLDT_DEFAULT_SIMILARITY_MEASURES, )),
    "STAT": (STAT, ()),
    "STAT_SFLDT": (STAT_SFLDT, (SFLDT_DEFAULT_SIMILARITY_MEASURES, )),
    "BARINEL": (BARINEL, ()),
    "STAT_BARINEL": (STAT_BARINEL, ()),
    "BARINEL_Paths": (BARINEL_Paths, ()),
    "BARINEL_Paths_After": (BARINEL_Paths_After, ()),
    "BARINEL_Paths_Difference": (BARINEL_Paths_Difference, ()),
    "STAT_BARINEL_Paths_After": (STAT_BARINEL_Paths_After, ()),
    "STAT_BARINEL_Paths_Difference": (STAT_BARINEL_Paths_Difference, ()),
    "BARINEL_Features": (BARINEL_Features, ()),
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
    if not isinstance(diagnoser_name, str):
        try:
            diagnoser_name = diagnoser_name[0]
        except:
            diagnoser_name = f"Failed to process diagnoser_name: {diagnoser_name}"
    assert diagnoser_name in diagnosers_dict, f"Diagnoser {diagnoser_name} is not supported"

    diagnoser_class, diagnoser_default_parameters = diagnosers_dict[diagnoser_name]
    if diagnoser_parameters is None:
        diagnoser_parameters = diagnoser_default_parameters
    return diagnoser_class, diagnoser_parameters