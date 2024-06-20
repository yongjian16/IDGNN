R"""
"""
#
import torch
import numpy as onp
import numpy.typing as onpt
from typing import List


def autocast(array: onpt.NDArray[onp.generic], /) -> onpt.NDArray[onp.generic]:
    R"""
    Automatically cast type for transfering.
    """
    #
    if onp.issubdtype(array.dtype, onp.integer):
        # For integer tensor.
        return array.astype(onp.int64)
    elif onp.issubdtype(array.dtype, onp.floating):
        # For float tensor.
        return array.astype(onp.float32)
    else:
        # UNEXPECT:
        # We should only have integer and float tensors.
        # TODO:
        # Support bool tensor.
        raise NotImplementedError(
            "Computation-device-transferable memory object must be integer or "
            "float tensors.",
        )


def transfer(
    local: List[onpt.NDArray[onp.generic]], device: str,
    /,
) -> List[torch.Tensor]:
    R"""
    Transfer local memory to remote device memory.
    """
    #
    return [torch.from_numpy(autocast(array)).to(device) for array in local]