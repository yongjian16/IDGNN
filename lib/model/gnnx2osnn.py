R"""
"""
#
import torch
import time
from typing import List, cast
from .model import Model
from .activate import activatize
from .gnnx2 import graphicalize
from .snn import sequentialize
from .initialize import glorot


class GNNx2oSNN(Model):
    R"""
    Graph neural network (2-layer) then sequential neural network.
    """
    def __init__(
        self,
        feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int, embed_inside_size: int,
        /,
        *,
        convolve: str, reduce: str, skip: bool, activate: str, concat: bool,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        #
        self.gnnx2 = (
            graphicalize(
                convolve, feat_input_size_edge, feat_input_size_node,
                embed_inside_size, embed_inside_size,
                skip=skip, activate=activate, concat=concat,
            )
        )
        self.snn = sequentialize(reduce, embed_inside_size, feat_target_size)
        self.activate = activatize(activate)

        #
        self.feat_target_size = feat_target_size

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        resetted = resetted + self.gnnx2.reset(rng)
        resetted = resetted + glorot(self.snn, rng)
        return resetted

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
        total_time_begin = time.time()
        graph_time = 0.0
        total_edges = 0

        #
        node_embeds: torch.Tensor

        # Must explicitly use built-in slice function to pass static
        # typing.
        (_, num_times) = edge_ranges.shape

        #
        graph_time_begin = time.time()
        snapshots = (
            torch.stack(
                [
                    self.gnnx2.forward(
                        edge_tuples[
                            :, slice(edge_ranges[0, t], edge_ranges[0, t]),
                        ],
                        edge_feats[
                            slice(edge_ranges[0, t], edge_ranges[0, t]),
                        ],
                        node_feats[:, :, t],
                    )
                    for t in range(num_times)
                ]
            )
        )
        graph_time = graph_time + (time.time() - graph_time_begin)
        total_edges = (
            total_edges
            + sum(
                cast(int, (edge_ranges[1, t] - edge_ranges[0, t]).item())
                for t in range(num_times)
            )
        )

        #
        (node_embeds, _) = self.snn.forward(self.activate(snapshots))
        total_time = time.time() - total_time_begin
        self.COSTS["graph"].append(graph_time)
        self.COSTS["non-graph"].append(total_time - graph_time)
        self.COSTS["edges"].append(total_edges)
        return node_embeds[-1]