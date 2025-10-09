import os
import importlib
import inspect

from .ADiagnoser import ADiagnoser
from .Oracle import Oracle

DIAGNOSER_CLASSES_DICT = {}

def _load_diagnoser_classes():
    """
    Load all classes in this package and subpackages that inherit from ADiagnoser.
    """
    base_dir = os.path.dirname(__file__)
    base_pkg = __name__

    for dirpath, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith(".py") and filename != "__init__.py":
                # Build module path
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, base_dir)
                module_name = rel_path[:-3].replace(os.sep, '.')
                full_module_name = f"{base_pkg}.{module_name}"

                module = importlib.import_module(full_module_name)
                for _, cls in inspect.getmembers(
                    module,
                    lambda c: inspect.isclass(c) and issubclass(c, ADiagnoser) and c is not ADiagnoser
                ):
                    DIAGNOSER_CLASSES_DICT[cls.__name__] = cls

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

__all__ = ["ADiagnoser", "get_diagnoser", "Oracle"]