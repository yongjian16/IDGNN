R"""
"""
#
import torch
from typing import List, Tuple


#
MSE = 0
RMSE = 1
MAPE = 2


def mse_loss(output: torch.Tensor, target: torch.Tensor, /) -> torch.Tensor:
    R"""
    Loss function.
    """
    #
    if not len(output.shape) == len(target.shape) == 2:
        # UNEXPECT:
        # Loss function must be computed on batched vectors.
        raise NotImplementedError(
            "Loss function must be computed on batched vectors.",
        )

    #
    return torch.mean(torch.mean((output - target) ** 2, dim=1))


def metrics(
    output: torch.Tensor, target: torch.Tensor,
    /,
) -> List[Tuple[int, float]]:
    R"""
    All evaluation metrics.
    """
    #
    if not len(output.shape) == len(target.shape) == 2:
        # UNEXPECT:
        # Loss function must be computed on batched vectors.
        raise NotImplementedError(
            "Loss function must be computed on batched vectors.",
        )

    #
    mse = torch.mean((output - target) ** 2, dim=1)
    rmse = torch.sqrt(mse)
    mape = (
        torch.mean(torch.abs(output - target) / (torch.abs(target) + 1), dim=1)
    )
    return (
        [
            (len(mse), torch.sum(mse).item()),
            (len(rmse), torch.sum(rmse).item()),
            (len(mape), torch.sum(mape).item()),
        ]
    )