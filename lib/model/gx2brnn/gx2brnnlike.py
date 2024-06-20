R"""
"""
#
import abc
import torch
from ..model import Model
from ..gnnx2 import graphicalize


class Gx2bRNNLike(Model):
    R"""
    Graph (2-layer) over branch RNN-like.
    """
    #
    NUM_KERNELS: int

    def __init__(
        self,
        feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int, embed_inside_size: int,
        /,
        *,
        convolve: str, skip: bool, activate: str, concat: bool,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        #
        self.gnnx2_i = (
            torch.nn.ModuleList(
                [
                    graphicalize(
                        convolve, feat_input_size_edge, feat_input_size_node,
                        embed_inside_size, embed_inside_size,
                        skip=skip, activate=activate, concat=concat,
                    )
                    for _ in range(self.NUM_KERNELS)
                ],
            )
        )
        self.gnnx2_h = (
            torch.nn.ModuleList(
                [
                    graphicalize(
                        convolve, feat_input_size_edge, embed_inside_size,
                        embed_inside_size, embed_inside_size,
                        skip=skip, activate=activate, concat=concat,
                    )
                    for _ in range(self.NUM_KERNELS)
                ],
            )
        )

        #
        self.rec_input_size = embed_inside_size

        #
        self.feat_target_size = feat_target_size

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        resetted = (
            resetted
            + sum(self.gnnx2_i[k].reset(rng) for k in range(self.NUM_KERNELS))
        )
        resetted = (
            resetted
            + sum(self.gnnx2_h[k].reset(rng) for k in range(self.NUM_KERNELS))
        )
        return resetted

    @abc.abstractmethod
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
        ...