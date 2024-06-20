R"""
"""
#
import numpy as onp
import numpy.typing as onpt
from typing import Tuple, Union
from ..stats.shape import shape as shape0


def shape5(shape: str, size: int, dtype: str, meaningless: bool, /) -> str:
    R"""
    Get visible shape of given array.
    """
    #
    if meaningless:
        #
        return "\x1b[90m{:s} ({:d}) {:s}*\x1b[0m".format(shape, size, dtype)
    else:
        #
        return "{:s} ({:d}) {:s}".format(shape, size, dtype)


def shape(
    array: onpt.NDArray[onp.generic],
    /,
) -> Tuple[str, Tuple[str, int, str, bool]]:
    R"""
    Get visible shape of given array.
    """
    (shape, size, dtype, meaningless) = shape0(array)
    string = shape5(shape, size, dtype, meaningless)
    return (string, (shape, size, dtype, meaningless))