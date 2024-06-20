#
import torch
import torch_geometric as thgeo
import torch_scatter as thidx
import time
from typing import cast
from .model import Model
from .initialize import glorot
from .activate import activatize
from .snn import auto_num_heads, sequentialize


class TGNUniMP(torch.nn.Module):
    R"""
    TGN attention convolution (by UniMP rather than TGAT).
    """
    def __init__(
        self,
        feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int, time_encoder: torch.nn.Module,
        /,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        torch.nn.Module.__init__(self)

        #
        heads = auto_num_heads(feat_target_size)
        self.time_encoder = time_encoder
        self.gnn = (
            thgeo.nn.TransformerConv(
                feat_input_size_node, feat_target_size // heads,
                heads=heads,
                edge_dim=(
                    feat_input_size_edge + cast(int, time_encoder.out_channels)
                ),
            )
        )

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        resetted = resetted + glorot(self.gnn.lin_key, rng)
        resetted = resetted + glorot(self.gnn.lin_query, rng)
        resetted = resetted + glorot(self.gnn.lin_value, rng)
        resetted = resetted + glorot(self.gnn.lin_edge, rng)
        resetted = resetted + glorot(self.gnn.lin_skip, rng)
        return resetted

    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        edge_times_rel: torch.Tensor, node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        node_embeds: torch.Tensor

        #
        edge_timefeats = (
            self.time_encoder.forward(
                torch.reshape(edge_times_rel, (len(edge_times_rel), 1)),
            )
        )
        edge_feats = torch.cat((edge_timefeats, edge_feats), dim=1)
        node_embeds = self.gnn.forward(node_feats, edge_tuples, edge_feats)
        return node_embeds


class TGNOptimx2(Model):
    R"""
    Temporal graph network (optimized) (2-convolution-layer).
    """
    def __init__(
        self,
        feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int, embed_inside_size: int,
        /,
        *,
        convolve: str, reduce_mem: str, reduce_node: str, skip: bool,
        activate: str, feat_timestamp_axis: int,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        #
        if convolve != "unimp":
            # UNEXPECT:
            # TGN only supports TGAT as memory convolution.
            raise NotImplementedError(
                "TGN (PyG) only supports UniMP as memory convolution.",
            )
        if reduce_mem != "gru":
            # UNEXPECT:
            # TGN subclasses require GRU memory reducer.
            raise NotImplementedError(
                "TGN (PyG) requires GRU sequential memory reducer.",
            )
        if feat_timestamp_axis != feat_input_size_node:
            # UNEXPECT:
            # TGN requires the last node feature to be timestamp.
            raise NotImplementedError(
                "Timestamped dynamic graph model (TGAT, TGN) requires the "
                "last (automatically extend in batching) node feature axis "
                "to be timestamp axis.",
            )

        #
        self.mem_size = embed_inside_size
        self.timeaxis = feat_timestamp_axis

        # TGN memory does not utilize raw node features (paper equation 1).
        # We need an extra transformation to fit edge and node message size.
        self.time_encoder = thgeo.nn.models.tgn.TimeEncoder(embed_inside_size)
        self.snn_mem = (
            sequentialize(
                reduce_mem + "[]",
                self.mem_size * 2 + self.time_encoder.out_channels
                + feat_input_size_edge,
                embed_inside_size,
            )
        )
        self.mem_transform_node = (
            torch.nn.Linear(
                self.mem_size + self.time_encoder.out_channels
                + feat_input_size_node,
                self.mem_size * 2 + self.time_encoder.out_channels
                + feat_input_size_edge,
            )
        )
        self.snn_node = (
            sequentialize(reduce_node, feat_input_size_node, embed_inside_size)
        )

        #
        self.tgnn1 = (
            TGNUniMP(
                feat_input_size_edge, embed_inside_size, embed_inside_size,
                self.time_encoder,
            )
        )
        self.tgnn2 = (
            TGNUniMP(
                feat_input_size_edge, embed_inside_size, feat_target_size,
                self.time_encoder,
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
        resetted = resetted + glorot(self.snn_mem, rng)
        resetted = resetted + glorot(self.mem_transform_node, rng)
        resetted = resetted + glorot(self.snn_node, rng)
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
        (edge_times_rel, edge_times_inc) = edge_times
        node_times_inc = node_times[:, 1]

        #
        num_nodes = len(node_masks)
        (_, num_times) = edge_ranges.shape

        # Get TGN memory states.
        # We use the most recent messsage aggregator which is the best in
        # original paper.
        # Since the whole batch contains full sequence, we will use relative
        # timestamp instead of timestamp offset for better efficency.
        node_states = (
            torch.zeros(
                (num_nodes, self.mem_size),
                dtype=node_feats.dtype, device=node_feats.device,
            )
        )
        node_memids = torch.arange(num_nodes, device=node_feats.device)
        for t in range(num_times):
            # Get edge slice for current snapshot.
            # Must explicitly use built-in slice function to pass static
            # typing.
            edge_snap_slice = slice(edge_ranges[0, t], edge_ranges[1, t])

            # Get edge messages.
            # In message, we use incremental timestamp data.
            edge_snap_src_states = node_states[edge_tuples[0, edge_snap_slice]]
            edge_snap_dst_states = node_states[edge_tuples[1, edge_snap_slice]]
            edge_snap_feats = edge_feats[edge_snap_slice]
            edge_snap_times = edge_times_inc[edge_snap_slice]
            edge_snap_timefeats = (
                self.time_encoder.forward(
                    torch.reshape(edge_snap_times, (len(edge_snap_times), 1)),
                )
            )
            edge_snap_messages = (
                torch.cat(
                    (
                        edge_snap_src_states, edge_snap_dst_states,
                        edge_snap_feats, edge_snap_timefeats,
                    ),
                    dim=1,
                )
            )

            # Get node messages.
            # Assume source and destiniation nodes are both the node itself.
            # In message, we use incremental timestamp data.
            node_snap_states = node_states
            node_snap_feats = node_feats[:, :, t]
            node_snap_times = node_times_inc[:, t]
            node_snap_timefeats = (
                self.time_encoder.forward(
                    torch.reshape(node_snap_times, (len(node_snap_times), 1)),
                )
            )
            node_snap_messages = (
                torch.cat(
                    (node_snap_states, node_snap_feats, node_snap_timefeats),
                    dim=1,
                )
            )
            node_snap_messages = (
                self.mem_transform_node.forward(node_snap_messages)
            )

            # Collect all edge and node messages together and keep only the
            # most recent messages for each destination node.
            snap_messages = (
                torch.cat((edge_snap_messages, node_snap_messages), dim=0)
            )
            message_dsts = (
                torch.cat((edge_tuples[1, edge_snap_slice], node_memids))
            )
            message_times = torch.cat((edge_snap_times, node_snap_times))
            (_, message_indices) = (
                thidx.scatter_min(message_times, message_dsts)
            )
            node_states = (
                self.snn_mem.forward(
                    snap_messages[message_indices], node_states,
                )
            )

        # Get node transformed embeddings.
        (node_embeds, _) = (
            self.snn_node.forward(torch.permute(node_feats, (2, 0, 1)))
        )
        node_embeds = node_embeds[-1] + node_states

        # Convolve.
        # In convolution, we use relative timestamp data.
        graph_time_begin = time.time()
        node_embeds = (
            self.tgnn1.forward(
                edge_tuples, edge_feats, edge_times_rel, node_embeds,
            )
        )
        node_embeds = (
            self.tgnn2.forward(
                edge_tuples, edge_feats, edge_times_rel,
                self.activate(node_embeds),
            )
        )
        graph_time = graph_time + (time.time() - graph_time_begin)
        total_edges = total_edges + edge_tuples.shape[1]
        total_time = time.time() - total_time_begin
        self.COSTS["graph"].append(graph_time)
        self.COSTS["non-graph"].append(total_time - graph_time)
        self.COSTS["edges"].append(total_edges)
        return node_embeds