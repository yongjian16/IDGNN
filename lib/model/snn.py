R"""
"""
#
import torch
import numpy as onp
import more_itertools as xitertools
from typing import Tuple, cast


class Linear(torch.nn.Module):
    R"""
    Linear but recurrent module.
    """
    def __init__(self, feat_input_size: int, feat_target_size: int, /) -> None:
        R"""
        Initialize the class.
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.feat_input_size = feat_input_size
        self.feat_target_size = feat_target_size
        self.lin = torch.nn.Linear(self.feat_input_size, self.feat_target_size)

    def forward(
        self,
        tensor: torch.Tensor,
        /,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        R"""
        Forward.
        """
        #
        (num_times, num_samples, _) = tensor.shape
        return (
            torch.reshape(
                self.lin.forward(
                    torch.reshape(
                        tensor,
                        (num_times * num_samples, self.feat_input_size),
                    ),
                ),
                (num_times, num_samples, self.feat_target_size),
            ),
            tensor[-1],
        )


class Static(torch.nn.Module):
    R"""
    Treate static feature as dynamic.
    """
    def forward(
        self,
        tensor: torch.Tensor,
        /,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        R"""
        Forward.
        """
        #
        return (torch.reshape(tensor, (1, *tensor.shape)), tensor)


def auto_num_heads(embed_size: int, /) -> int:
    R"""
    Automatically get number of multi-heads.
    """
    #
    return (
        xitertools.first_true(
            range(int(onp.ceil(onp.sqrt(embed_size))), 0, -1),
            default=1, pred=lambda x: embed_size % x == 0 and x & (x - 1) == 0,
        )
    )


class MultiheadAttention(torch.nn.Module):
    R"""
    Multi-head attention with recurrent-like forward.
    """
    def __init__(self, feat_input_size: int, feat_target_size: int, /) -> None:
        R"""
        Initialize the class.
        """
        #
        torch.nn.Module.__init__(self)

        #
        embed_size = feat_target_size
        self.num_heads = auto_num_heads(embed_size)
        self.mha = torch.nn.MultiheadAttention(embed_size, self.num_heads)

        #
        self.transform: torch.nn.Module

        #
        if feat_input_size != embed_size:
            #
            self.transform = (
                torch.nn.Linear(feat_input_size, embed_size, bias=False)
            )
        else:
            #
            self.transform = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        R"""
        Forward.
        """
        #
        x = self.transform(x)
        (y, attn) = self.mha.forward(x, x, x)
        return (y, cast(torch.Tensor, attn))


def sequentialize(
    name: str, feat_input_size: int, feat_target_size: int,
    /,
) -> torch.nn.Module:
    R"""
    Get sequential module.
    """
    #
    if name == "linear":
        #
        return Linear(feat_input_size, feat_target_size)
    elif name == "gru":
        #
        return torch.nn.GRU(feat_input_size, feat_target_size)
    elif name == "lstm":
        #
        return torch.nn.LSTM(feat_input_size, feat_target_size)
    elif name == "gru[]":
        #
        return torch.nn.GRUCell(feat_input_size, feat_target_size)
    elif name == "lstm[]":
        #
        return torch.nn.LSTMCell(feat_input_size, feat_target_size)
    elif name == "mha":
        #
        return MultiheadAttention(feat_input_size,feat_target_size)
    elif name == "static":
        #
        return Static()
    else:
        # EXPECT:
        # It is possible to require unsupporting sequential model.
        raise RuntimeError(
            "Sequential module identifier \"{:s}\" is not supported."
            .format(name),
        )