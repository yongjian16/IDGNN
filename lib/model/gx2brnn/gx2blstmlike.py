R"""
"""
#
import torch
import numpy as onp
import time
from .gx2brnnlike import Gx2bRNNLike


class Gx2bLSTMLike(Gx2bRNNLike):
    R"""
    Graph (2-layer) over branch LSTM-like.
    """
    #
    NUM_KERNELS = 4
    INPUT = 0
    FORGET = 1
    CELL = 2
    OUTPUT = 3

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
        Gx2bRNNLike.__init__(
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
        resetted = Gx2bRNNLike.reset(self, rng)

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
        node_snap_cells = (
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

            #
            graph_time_begin = time.time()
            snap_inputs = (
                torch.sigmoid(
                    self.gnnx2_i[self.INPUT](
                        edge_snap_tuples, edge_snap_feats, node_snap_feats,
                    )
                    + (
                        self.gnnx2_h[self.INPUT](
                            edge_snap_tuples, edge_snap_feats,
                            node_snap_embeds,
                        )
                    )
                    + self.bias[self.INPUT],
                )
            )
            snap_forgets = (
                torch.sigmoid(
                    self.gnnx2_i[self.FORGET](
                        edge_snap_tuples, edge_snap_feats, node_snap_feats,
                    )
                    + (
                        self.gnnx2_h[self.FORGET](
                            edge_snap_tuples, edge_snap_feats,
                            node_snap_embeds,
                        )
                    )
                    + self.bias[self.FORGET],
                )
            )
            snap_cells = (
                torch.tanh(
                    self.gnnx2_i[self.CELL](
                        edge_snap_tuples, edge_snap_feats, node_snap_feats,
                    )
                    + (
                        self.gnnx2_h[self.CELL](
                            edge_snap_tuples, edge_snap_feats,
                            node_snap_embeds,
                        )
                    )
                    + self.bias[self.CELL],
                )
            )
            snap_outputs = (
                torch.sigmoid(
                    self.gnnx2_i[self.OUTPUT](
                        edge_snap_tuples, edge_snap_feats, node_snap_feats,
                    )
                    + (
                        self.gnnx2_h[self.OUTPUT](
                            edge_snap_tuples, edge_snap_feats,
                            node_snap_embeds,
                        )
                    )
                    + self.bias[self.OUTPUT],
                )
            )
            graph_time = graph_time + (time.time() - graph_time_begin)
            total_edges = total_edges + edge_snap_tuples.shape[1]

            #
            node_snap_cells = (
                snap_forgets * node_snap_cells + snap_inputs * snap_cells
            )
            node_snap_embeds = snap_outputs * torch.tanh(node_snap_cells)
        total_time = time.time() - total_time_begin
        self.COSTS["graph"].append(graph_time)
        self.COSTS["non-graph"].append(total_time - graph_time)
        self.COSTS["edges"].append(total_edges)
        return node_snap_embeds