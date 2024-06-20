R"""
"""
#
import torch
from typing import List
from .model import Model


class MovingWindow(Model):
    R"""
    Moving window.
    """
    def moveon(self, axis: List[int]) -> None:
        R"""
        Settile the axis to apply moving window.
        """
        #
        self.axis = axis

        #
        self.feat_target_size = len(axis)

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        # This is a non-parametric model.
        return 0


class MovingAverage(MovingWindow):
    R"""
    Moving average.
    """
    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        edge_ranges: torch.Tensor, edge_times: torch.Tensor,
        node_feats: torch.Tensor, node_times: torch.Tensor,
        node_masks: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        return torch.mean(node_feats[:, self.axis], dim=2)


class MovingLast(MovingWindow):
    R"""
    Moving Last.
    """
    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        edge_ranges: torch.Tensor, edge_times: torch.Tensor,
        node_feats: torch.Tensor, node_times: torch.Tensor,
        node_masks: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        return node_feats[:, self.axis, -1]