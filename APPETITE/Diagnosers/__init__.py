import os
import importlib
import inspect

from .ADiagnoser import ADiagnoser

DIAGNOSER_CLASSES_DICT = {}

def _load_diagnoser_classes():
    """
    Load all classes in the current module that inherit from ADiagnoser.
    """
    for module_name, _ in filter(lambda filename_splitted: len(filename_splitted) == 2 and filename_splitted[1] == "py" and filename_splitted[0] != "__init__",
                                 map(lambda filename: filename.split('.'),
                                     os.listdir(os.path.dirname(__file__))
                                     )):
        module = importlib.import_module(f".{module_name}", package=__name__)
        for _, module_class in inspect.getmembers(module, lambda module_class: inspect.isclass(module_class) and issubclass(module_class, ADiagnoser) and module_class is not ADiagnoser):
            DIAGNOSER_CLASSES_DICT[module_class.__name__] = module_class
        
_load_diagnoser_classes()

def get_diagnoser(diagnoser_name: str
 ) -> ADiagnoser:
    """
    Get a diagnoser class by its name.
    
    Parameters:
    diagnoser_name (str): The name of the diagnoser class.
    
    Returns:
    ADiagnoser: The diagnoser class.
    
    Raises:
    ValueError: If the diagnoser class is not found.
    """
    assert diagnoser_name in DIAGNOSER_CLASSES_DICT, f"Diagnoser {diagnoser_name} is not supported"
    
    return DIAGNOSER_CLASSES_DICT[diagnoser_name]

__all__ = ["ADiagnoser", "get_diagnoser"]