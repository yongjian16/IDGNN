R"""
"""
#
import torch
import numpy as onp
import time
from .gx2crnnlike import Gx2cRNNLike


class Gx2cGRULike(Gx2cRNNLike):
    R"""
    Graph (2-layer) over concatentation GRU-like.
    """
    #
    NUM_KERNELS = 3
    RESET = 0
    UPDATE = 1
    CELL = 2

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
        Gx2cRNNLike.__init__(
            self, feat_input_size_edge, feat_input_size_node, feat_target_size,
            embed_inside_size,
            convolve=convolve, skip=skip, activate=activate, concat=concat,
        )

        # Since we can not utilize linear transfomation trick in graph
        # recurrent neural network, we will not concatenate recurrent bias
        # parameters as common recurrent neural networks.
        self.rec_target_size = feat_target_size
        self.bias = (
            torch.nn.parameter.Parameter(
                torch.zeros(self.NUM_KERNELS, self.rec_target_size),
            )
        )

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = Gx2cRNNLike.reset(self, rng)

        #
        a = onp.sqrt(6 / (self.rec_input_size + self.rec_target_size))
        self.bias.data.uniform_(-a, a, generator=rng)
        resetted = resetted + self.bias.numel()
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
        num_nodes = len(node_masks)
        (_, num_times) = edge_ranges.shape

        #
        node_snap_embeds = (
            torch.zeros(
                (num_nodes, self.rec_target_size),
                dtype=node_feats.dtype, device=node_feats.device,
            )
        )
        for t in range(num_times):
            # Get edge slice for current snapshot.
            # Must explicitly use built-in slice function to pass static
            # typing.
            edge_snap_slice = slice(edge_ranges[0, t], edge_ranges[1, t])

            #
            edge_snap_tuples = edge_tuples[:, edge_snap_slice]
            edge_snap_feats = edge_feats[edge_snap_slice]
            node_snap_feats = node_feats[:, :, t]
            node_snap_concats = (
                torch.cat((node_snap_feats, node_snap_embeds), dim=1)
            )

            #
            graph_time_begin = time.time()
            snap_resets = (
                torch.sigmoid(
                    self.gnnx2[self.RESET](
                        edge_snap_tuples, edge_snap_feats, node_snap_concats,
                    )
                    + self.bias[self.RESET],
                )
            )
            snap_updates = (
                torch.sigmoid(
                    self.gnnx2[self.UPDATE](
                        edge_snap_tuples, edge_snap_feats, node_snap_concats,
                    )
                    + self.bias[self.UPDATE],
                )
            )
            snap_cells = (
                torch.tanh(
                    self.gnnx2[self.CELL](
                        edge_snap_tuples, edge_snap_feats,
                        torch.cat(
                            (node_snap_feats, snap_resets * node_snap_embeds),
                            dim=1,
                        )
                    )
                    + self.bias[self.CELL],
                )
            )
            graph_time = graph_time + (time.time() - graph_time_begin)
            total_edges = total_edges + edge_snap_tuples.shape[1]

            #
            node_snap_embeds = (
                snap_updates * node_snap_embeds
                + (1 - snap_updates) * snap_cells
            )
        total_time = time.time() - total_time_begin
        self.COSTS["graph"].append(graph_time)
        self.COSTS["non-graph"].append(total_time - graph_time)
        self.COSTS["edges"].append(total_edges)
        return node_snap_embeds