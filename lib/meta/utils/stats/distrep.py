R"""
"""
#
import numpy as onp
import numpy.typing as onpt
from typing import Tuple, Union, List, cast
from .setlike import SETLIKE, setlike
from .boxlike import BOXLIKE, boxlike


def distrep(
    array: onpt.NDArray[onp.generic],
    /,
    *,
    axis: Union[int, Tuple[int, ...]],
) -> Union[List[SETLIKE], List[BOXLIKE]]:
    R"""
    Get data distribution statistics of given array.
    """
    #
    if onp.issubdtype(array.dtype, onp.integer):
        #
        return setlike(array, axis=axis)
    elif onp.issubdtype(array.dtype, onp.floating):
        #
        return boxlike(array, axis=axis)
    else:
        # EXPECT:
        # It is possible to have improper arguments.
        raise RuntimeError(
            "Data distribution only supports integer or float array."
        )