R"""
"""
#
import torch
import torch_geometric as thgeo
import time
from typing import cast
from ..model import Model
from ..activate import activatize
from ..initialize import glorot
from .tgat import TGATConv


class TGATx2(Model):
    R"""
    Temporal graph attention neural network (2-layer).
    """
    def __init__(
        self,
        feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int, embed_inside_size: int,
        /,
        *,
        convolve: str, skip: bool, activate: str, feat_timestamp_axis: int,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        #
        if convolve != "tgat":
            # UNEXPECT:
            # TGAT must use TGAT as convolution.
            raise NotImplementedError(
                "Timestamped dynamic graph model (TGAT, TGN) must use TGAT as "
                "convolution.",
            )
        if skip:
            # UNEXPECT:
            # TGAT does not support skip-connection.
            raise NotImplementedError("TGAT does not support skip-connection.")
        if feat_timestamp_axis != feat_input_size_node:
            # UNEXPECT:
            # TGN requires the last node feature to be timestamp.
            raise NotImplementedError(
                "Timestamped dynamic graph model (TGAT, TGN) requires the "
                "last (automatically extend in batching) node feature axis "
                "to be timestamp axis.",
            )

        #
        self.timeaxis = feat_timestamp_axis

        # Time encoder is shared by different layers.
        self.time_encoder = thgeo.nn.models.tgn.TimeEncoder(embed_inside_size)

        # Use invalid timestamp axis since we will deal with timestamp data
        # only once at block level for all layers in the same time.
        self.tgnn1 = (
            TGATConv(
                feat_input_size_edge, feat_input_size_node, embed_inside_size,
                feat_timestamp_axis=self.timeaxis,
                time_encoder=self.time_encoder,
            )
        )
        self.tgnn2 = (
            TGATConv(
                feat_input_size_edge, embed_inside_size, feat_target_size,
                feat_timestamp_axis=embed_inside_size,
                time_encoder=self.time_encoder,
            )
        )
        self.activate = activatize(activate)

        #
        self.feat_target_size = feat_target_size

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        resetted = resetted + glorot(self.time_encoder.lin, rng)
        resetted = resetted + self.tgnn1.reset(rng)
        resetted = resetted + self.tgnn2.reset(rng)
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

        # Split timestamp data.
        # We only need relative edge timestamps.
        # We will merge relative edge timestamp embeddings with edge features.
        edge_times_rel = edge_times[0]
        edge_timefeats = (
            self.time_encoder.forward(
                torch.reshape(edge_times_rel, (len(edge_times_rel), 1)),
            )
        )
        edge_feats = torch.cat((edge_feats, edge_timefeats), dim=1)

        # We need to shift source node IDs to differentiate them by timestamps.
        # Shift only source node so that we will still aggregate to the same
        # destination nodes for different timestamps.
        (num_nodes, num_feats, num_times) = node_feats.shape
        edge_srcs = (
            edge_tuples[0]
            + (
                torch.cat(
                    [
                        torch.full(
                            (
                                cast(
                                    int, edge_ranges[1, t] - edge_ranges[0, t],
                                ),
                            ),
                            t * num_nodes,
                            dtype=edge_tuples.dtype, device=edge_tuples.device,
                        )
                        for t in range(num_times)
                    ],
                )
            )
        )
        edge_dsts = edge_tuples[1]
        edge_tuples = torch.stack((edge_srcs, edge_dsts))

        # Transform temporal node features to fit shifted edge connectivities.
        # Node mask will not change.
        # We need a additional node ranges to differentiate source nodes of
        # different timestamps.
        node_feats = (
            torch.reshape(
                torch.permute(node_feats, (2, 0, 1)),
                (num_times * num_nodes, num_feats),
            )
        )
        node_ranges_src = (
            torch.arange(num_times + 1, device=node_feats.device) * num_nodes
        )
        node_ranges_src = (
            torch.stack((node_ranges_src[:-1], node_ranges_src[1:]))
        )

        # We only care about the final timestamp at the last layer.
        # For other layers, we need to forward for all timestamps.
        graph_time_begin = time.time()
        node_embeds = (
            self.tgnn1.forwardall(
                edge_tuples, edge_feats, edge_times, edge_ranges, node_feats,
                node_ranges_src, node_masks,
            )
        )
        node_embeds = (
            self.tgnn2.forwardfin(
                edge_tuples, edge_feats, edge_times, edge_ranges,
                self.activate(node_embeds), node_ranges_src, node_masks,
            )
        )
        graph_time = graph_time + (time.time() - graph_time_begin)
        total_edges = total_edges + edge_tuples.shape[1]
        total_time = time.time() - total_time_begin
        self.COSTS["graph"].append(graph_time)
        self.COSTS["non-graph"].append(total_time - graph_time)
        self.COSTS["edges"].append(total_edges)
        return node_embeds