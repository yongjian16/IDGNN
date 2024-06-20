R"""
"""
#
import numpy as onp
import numpy.typing as onpt
from typing import Tuple, Union, List


#
SETLIKE = Tuple[int, int, int, int]


def nunique(
    array: onpt.NDArray[onp.generic],
    /,
    *,
    axis: Union[int, Tuple[int, ...]], keepdims: bool,
) -> onpt.NDArray[onp.generic]:
    R"""
    NumPy uniqueness counting extension for generic axis.
    We assume it must be integer array.
    """
    # TODO:
    # Ensure that there is no popular public library.
    # Get essential shape info.
    ax = [axis] if isinstance(axis, int) else list(axis)
    finshape = onp.array(array.shape)
    actshape = onp.delete(finshape, ax)
    unishape = finshape[ax]
    finshape[ax] = 1
    if not keepdims:
        # Squeeze.
        finshape = finshape[finshape > 1]

    # Move active axes ahead, and flatten uniqueness computing axes.
    array = (
        onp.transpose(
            array,
            onp.concatenate((onp.delete(onp.arange(array.ndim), ax), ax)),
        )
    )
    array = onp.reshape(array, (*actshape, int(onp.prod(unishape))))

    # Count uniqueness.
    array = onp.sum(onp.diff(onp.sort(array)) != 0, axis=-1) + 1
    return onp.reshape(array, finshape)


def setlike(
    array: onpt.NDArray[onp.generic],
    /,
    *,
    axis: Union[int, Tuple[int, ...]],
) -> List[SETLIKE]:
    R"""
    Get set-like statistics of given array.
    """
    # Collect set statistics.
    vmins = onp.min(array, axis=axis, keepdims=True)
    vmaxs = onp.max(array, axis=axis, keepdims=True)
    vsizes = vmaxs - vmins + 1
    vuniques = nunique(array, axis=axis, keepdims=True)

    # Flatten.
    num = max(max(stats.shape) for stats in (vmins, vmaxs, vsizes, vuniques))
    vmins = onp.reshape(vmins, (num,))
    vmaxs = onp.reshape(vmaxs, (num,))
    vsizes = onp.reshape(vsizes, (num,))
    vuniques = onp.reshape(vuniques, (num,))

    # Collect statistics for each column.
    return (
        [
            (
                vmin.item(), vmax.item(), vsize.item(), vunique.item(),
            )
            for (vmin, vmax, vsize, vunique) in (
                zip(vmins, vmaxs, vsizes, vuniques)
            )
        ]
    )