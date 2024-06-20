R"""
"""
#
import torch


def grad_clip_norm2(model: torch.nn.Module, scale: float) -> None:
    R"""
    Clip gradient by norm 2.
    """
    #
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), scale,
        norm_type=2.0, error_if_nonfinite=True,
    )


def grad_clip_norminf(model: torch.nn.Module, scale: float) -> None:
    R"""
    Clip gradient by norm inf.
    """
    #
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), scale,
        norm_type=float("inf"), error_if_nonfinite=True,
    )


def grad_clip_value(model: torch.nn.Module, scale: float) -> None:
    R"""
    Clip gradient by value.
    """
    #
    torch.nn.utils.clip_grad_value_(model.parameters(), scale)


#
GRADCLIPS = {
    "norm2": grad_clip_norm2,
    "norminf": grad_clip_norminf,
    "value": grad_clip_value,
}