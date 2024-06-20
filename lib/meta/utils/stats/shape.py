R"""
"""
#
import numpy as onp
import numpy.typing as onpt
from typing import Tuple, Union


#
SHAPE = Tuple[str, int, str, bool]


def shape(array: onpt.NDArray[onp.generic], /) -> SHAPE:
    R"""
    Get shape of given array.
    """
    # All values being same is meaningless.
    shape = "x".join(str(dim) for dim in array.shape)
    size = array.size
    (_, dtype) = repr(array.dtype).split(".")
    meaningless = onp.min(array) == onp.max(array)
    return (shape, size, dtype, meaningless)