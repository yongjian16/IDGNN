R"""
"""
#
import numpy as onp
import numpy.typing as onpt
from typing import Tuple, List, Union

def standardize(array: onpt.NDArray[onp.generic], fromarray: onpt.NDArray[onp.generic], 
                /, 
                *, 
                axis: Union[int, Tuple[int, ...]],) -> onpt.NDArray[onp.generic]:
    # Collect normalization factors from fromarray.
    means = onp.mean(fromarray, axis=axis, keepdims=True)
    stds = onp.std(fromarray, axis=axis, keepdims=True)
    stds[stds == 0] = 1

    # Normalize given array by collected factors.
    array = (array - means) / stds

    # Format collected factors.
    num = max(means.shape)
    means = onp.reshape(means, (num,))
    stds = onp.reshape(stds, (num,))

    factors = (
        [(mean.item(), std.item()) for mean, std in zip(means, stds)]
    )

    return (array, factors)
    

def normalize(
    array: onpt.NDArray[onp.generic], fromarray: onpt.NDArray[onp.generic],
    /,
    *,
    axis: Union[int, Tuple[int, ...]]
) -> Tuple[onpt.NDArray[onp.generic], List[Tuple[float, float]]]:
    R"""
    Normalize array only on a specific axis.
    """
    # Safety check.
    if len(array.shape) != len(fromarray.shape):
        # EXPECT:
        # It is possible to have improper arguments.
        raise RuntimeError(
            "Normalizing array should have the same number of dimensions as "
            "normalizing factor provider array."
        )

    # Collect normalization factors from fromarray.
    vmins = onp.min(fromarray, axis=axis, keepdims=True)
    vmaxs = onp.max(fromarray, axis=axis, keepdims=True)
    vsizes = vmaxs - vmins
    vsizes[vsizes == 0] = 1

    # Normalize given array by collected factors.
    array = (array - vmins) / vsizes

    # Format collected factors.
    num = max(vsizes.shape)
    vmins = onp.reshape(vmins, (num,))
    vmaxs = onp.reshape(vmaxs, (num,))
    vsizes = onp.reshape(vsizes, (num,))
    factors = (
        [(vmin.item(), vsize.item()) for vmin, vsize in zip(vmins, vsizes)]
    )
    return (array, factors)