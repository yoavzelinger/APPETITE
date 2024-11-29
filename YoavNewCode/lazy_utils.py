from collections.abc import Iterable
from typing import Generator, Callable

def lazy_product(
        generators: list[Callable[[], Generator[object, object, None]]], 
        args_list: object | list[object] = [],
        current_product=[]
 ) -> Generator[list[object], None, None]:
    """
    Generate the cartesian product of the given generators.
    The function is lazy and generates the product on the fly.

    Parameters:
        generators (list[Callable[[], Generator[object, object, None]]): List of generators.
        args_list (object | list[object]): single argument or list of arguments for the generators.
            The argument list is all of the arguments to be passed to each generator (not one argument per generator).
            FOR NO ARGUMENTS DON'T PASS ANYTHING (NO NONE!).
        current_product (list[object]): The current product.
    """
    if not generators:
        yield current_product
    else:
        if not isinstance(args_list, Iterable):
            # Wrap with list so can be unpacked
            args_list = [] if args_list is None else [args_list]
        for item in generators[0](*args_list):
            yield from lazy_product(generators[1:], args_list, current_product + [item])
