#
import torch
import torch_geometric as thgeo
from .model import Model
from .initialize import glorot
from .activate import activatize
from .snn import sequentialize
from .tgnoptimx2 import TGNUniMP


class TGNMemory(thgeo.nn.TGNMemory):
    """
    TGN memory.
    Initial last updated indices type is fixed.
    """
    def __reset_message_store__(self, /) -> None:
        R"""
        Reset memory storage.
        """
        #
        i = self.memory.new_empty((0,), dtype=torch.long)
        t = self.memory.new_empty((0,), dtype=self.last_update.dtype)
        msg = self.memory.new_empty((0, self.raw_msg_dim))

        #
        self.msg_s_store = {j: (i, i, t, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: (i, i, t, msg) for j in range(self.num_nodes)}


class TGNx2(Model):
    R"""
    Temporal graph network (2-convolution-layer).
    """
    def __init__(
        self,
        feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int, embed_inside_size: int,
        /,
        *,
        convolve: str, reduce_mem: str, reduce_node: str, skip: bool,
        activate: str, feat_timestamp_axis: int, num_nodes: int,
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

        # Create raw TGN with last updated indices type being fixed.
        self.memory = (
            TGNMemory(
                num_nodes, feat_input_size_edge, embed_inside_size,
                embed_inside_size,
                message_module=(
                    thgeo.nn.models.tgn.IdentityMessage(
                        feat_input_size_edge, embed_inside_size,
                        embed_inside_size,
                    )
                ),
                aggregator_module=thgeo.nn.models.tgn.LastAggregator(),
            )
        )
        del self.memory.last_update
        self.memory.register_buffer(
            "last_update",
            torch.empty(
                num_nodes,
                dtype=self.memory.time_enc.lin.weight.data.dtype,
            ),
        )

        # TGN memory does not utilize raw node features (paper equation 1).
        # We need an extra transformation to fit edge and node message size.
        self.time_encoder = self.memory.time_enc
        self.snn_mem = self.memory.gru
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
        node_embeds: torch.Tensor

        # Split timestamp data.
        # We only have relative timestamps w.r.t. corresponding last event for
        # each edge (descending).
        # TGN will automatically compute the incremental timestamps, thus we
        # need to use negative timestamps (ascending) so that timestamps are
        # incremental.
        (edge_times_rel,) = edge_times
        edge_times_asc = torch.max(edge_times_rel) - edge_times_rel

        #
        num_nodes = len(node_masks)
        (_, num_times) = edge_ranges.shape

        # Get TGN memory states.
        # We use the most recent messsage aggregator which is the best in
        # original paper.
        # Since the whole batch contains full sequence, we will use relative
        # timestamp instead of timestamp offset for better efficency.
        self.memory.reset_state()
        node_memids = torch.arange(num_nodes, device=node_feats.device)
        # \\ node_state_buf = []
        # \\ for t in range(num_times):
        # \\     # Get edge slice for current snapshot.
        # \\     # Must explicitly use built-in slice function to pass static
        # \\     # typing.
        # \\     edge_snap_slice = slice(edge_ranges[0, t], edge_ranges[1, t])
        # \\ 
        # \\     # We must do each timestamp one-by-one explicitly to keep essential
        # \\     # memory for backpropagation.
        # \\     edge_srcs = edge_tuples[0, edge_snap_slice]
        # \\     edge_dsts = edge_tuples[1, edge_snap_slice]
        # \\     edge_ts = edge_times_asc[edge_snap_slice]
        # \\     edge_msgs = edge_feats[edge_snap_slice]
        # \\     self.memory.update_state(edge_srcs, edge_dsts, edge_ts, edge_msgs)
        # \\     (node_states, _) = self.memory.forward(node_memids)
        # \\     node_state_buf.append(node_states)
        self.memory.update_state(edge_tuples[0], edge_tuples[1], edge_times_asc, edge_feats)
        (node_states, _) = self.memory.forward(node_memids)

        # Get node transformed embeddings.
        (node_embeds, _) = (
            self.snn_node.forward(torch.permute(node_feats, (2, 0, 1)))
        )
        node_embeds = node_embeds[-1] + node_states

        # Convolve.
        # In convolution, we use relative timestamp data.
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
        return node_embeds