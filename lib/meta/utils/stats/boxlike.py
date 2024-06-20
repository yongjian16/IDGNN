R"""
"""
#
import numpy as onp
import numpy.typing as onpt
from typing import Tuple, Union, List


#
BOXLIKE = Tuple[float, float, float, float, float, float, float]


def boxlike(
    array: onpt.NDArray[onp.generic],
    /,
    *,
    axis: Union[int, Tuple[int, ...]],
) -> List[BOXLIKE]:
    R"""
    Get box-like statistics of given array.
    """
    #
    if not onp.issubdtype(array.dtype, onp.floating):
        # EXPECT:
        # It is possible to have improper arguments.
        raise RuntimeError(
            "Can not collect box-like statistics for a non-float array.",
        )

    # Collect box statistics.
    vmins = onp.min(array, axis=axis, keepdims=False)
    vq1s = onp.quantile(array, 0.25, axis=axis, keepdims=False)
    vmedians = onp.median(array, axis=axis, keepdims=False)
    vmeans = onp.mean(array, axis=axis, keepdims=False)
    vstds = onp.std(array, axis=axis, keepdims=False)
    vq3s = onp.quantile(array, 0.75, axis=axis, keepdims=False)
    vmaxs = onp.max(array, axis=axis, keepdims=False)

    # Flatten.
    num = (
        max(
            1 if stats.ndim == 0 else max(stats.shape)
            for stats in (vmins, vq1s, vmedians, vmeans, vstds, vq3s, vmaxs)
        )
    )
    vmins = onp.reshape(vmins, (num,))
    vq1s = onp.reshape(vq1s, (num,))
    vmedians = onp.reshape(vmedians, (num,))
    vmeans = onp.reshape(vmeans, (num,))
    vstds = onp.reshape(vstds, (num,))
    vq3s = onp.reshape(vq3s, (num,))
    vmaxs = onp.reshape(vmaxs, (num,))

    # Collect statistics for each column.
    return (
        [
            (
                vmin.item(), vq1.item(), vmedian.item(), vmean.item(),
                vstd.item(), vq3.item(), vmax.item(),
            )
            for (vmin, vq1, vmedian, vmean, vstd, vq3, vmax) in (
                zip(vmins, vq1s, vmedians, vmeans, vstds, vq3s, vmaxs)
            )
        ]
    )