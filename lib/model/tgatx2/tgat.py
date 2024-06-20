R"""
"""
#
import torch
import torch_geometric as thgeo
from typing import List, Optional, cast
from ..model import Model
from ..initialize import glorot
from ..snn import auto_num_heads


class TGATConv(Model):
    R"""
    Temporal graph attention neural network.
    """
    def __init__(
        self,
        feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int,
        /,
        *,
        feat_timestamp_axis: int, time_encoder: Optional[torch.nn.Module],
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        #
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

        #
        self.time_encoder_shared = time_encoder is not None
        if self.time_encoder_shared:
            # Share time encoder implicitly.
            self.time_encoder = (
                cast(thgeo.nn.models.tgn.TimeEncoder, time_encoder)
            )
        else:
            # Create time encoder explicitly.
            self.time_encoder = (
                thgeo.nn.models.tgn.TimeEncoder(feat_target_size)
            )

        # Pay attention to remove timestamp axis from node input features.
        heads = auto_num_heads(feat_target_size)
        self.gnn = (
            thgeo.nn.GATConv(
                feat_input_size_node, feat_target_size // heads,
                heads=heads, edge_dim=feat_input_size_edge + feat_target_size,
            )
        )

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        if not self.time_encoder_shared:
            # Only initialize for non-sharing time encoder.
            resetted = resetted + glorot(self.time_encoder.lin, rng)
        resetted = resetted + glorot(self.gnn, rng)
        return resetted

    def forwardall(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        edge_times: torch.Tensor, edge_ranges: torch.Tensor,
        node_feats: torch.Tensor, node_ranges_src: torch.Tensor,
        node_masks: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward for all timestamps.
        """
        #
        node_embeds_buf: List[torch.Tensor]
        node_embeds: torch.Tensor

        #
        (_, num_times) = edge_ranges.shape
        num_nodes = len(node_masks)

        # Pay attention that GNN will generate outputs even for shifted source
        # nodes.
        # Thus, we need to explicitly truncate output of each step.
        node_embeds_buf = []
        for t in range(num_times):
            # Get source node and edge slice for current snapshot.
            # Must explicitly use built-in slice function to pass static
            # typing.
            node_snap_slice_src = slice(None, node_ranges_src[1, t])
            edge_snap_slice = slice(None, edge_ranges[1, t])

            #
            node_embeds_buf.append(
                self.gnn.forward(
                    node_feats[node_snap_slice_src],
                    edge_tuples[:, edge_snap_slice],
                    edge_feats[edge_snap_slice],
                )[:num_nodes]
            )
        node_embeds = torch.cat(node_embeds_buf)
        return node_embeds

    def forwardfin(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        edge_times: torch.Tensor, edge_ranges: torch.Tensor,
        node_feats: torch.Tensor, node_ranges: torch.Tensor,
        node_masks: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward for the final timestamp.
        """
        #
        node_embeds: torch.Tensor

        #
        num_nodes = len(node_masks)

        # Pay attention that GNN will generate outputs even for shifted source
        # nodes.
        # Thus, we need to explicitly truncate output of each step.
        node_embeds = (
            self.gnn.forward(node_feats, edge_tuples, edge_feats)[:num_nodes]
        )
        return node_embeds