R"""
"""
#
import numpy as onp
import numpy.typing as onpt
from typing import Tuple, Union, List
from ..stats.setlike import SETLIKE
from ..stats.boxlike import BOXLIKE
from .setlike import setlike
from .boxlike import boxlike


def distrep(
    array: onpt.NDArray[onp.generic],
    /,
    *,
    axis: Union[int, Tuple[int, ...]], n: int,
) -> Tuple[str, Union[List[SETLIKE], List[BOXLIKE]]]:
    R"""
    Get visible data distribution statistics of given array.
    """
    #
    if onp.issubdtype(array.dtype, onp.integer):
        #
        return setlike(array, axis=axis, n=n)
    elif onp.issubdtype(array.dtype, onp.floating):
        #
        return boxlike(array, axis=axis, n=n)
    else:
        # UNEXPECT:
        # Only distribution statistics over integer and float tensors are
        # defined.
        # TODO:
        # Support bool tensor.
        raise NotImplementedError(
            "Only distribution statistics over integer and float tensors are "
            "defined.",
        )