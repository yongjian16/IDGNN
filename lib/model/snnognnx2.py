R"""
"""
#
import torch
import time
from collections import OrderedDict
from .model import Model
from .activate import activatize
from .gnnx2 import graphicalize, GNNx2Concat
from .snn import sequentialize
from .initialize import glorot


class SNNoGNNx2(Model):
    R"""
    Sequential neural network then graph neural network (2-layer).
    """
    def __init__(
        self,
        feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int, embed_inside_size: int,
        /,
        *,
        convolve: str, reduce_edge: str, reduce_node: str, skip: bool,
        activate: str, concat: bool,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        #
        self.reduce_edge = reduce_edge
        self.reduce_node = reduce_node
        self.snn_edge = (
            sequentialize(reduce_edge, feat_input_size_edge, embed_inside_size)
        )
        self.snn_node = (
            sequentialize(reduce_node, feat_input_size_node, embed_inside_size)
        )
        self.gnnx2 = (
            graphicalize(
                convolve,
                (
                    feat_input_size_edge
                    if reduce_edge == "static" else
                    embed_inside_size
                ),
                embed_inside_size, feat_target_size, embed_inside_size,
                skip=skip, activate=activate, concat=concat,
            )
        )
        self.activate = activatize(activate)

        #
        self.feat_target_size = (
            feat_target_size + int(concat) * embed_inside_size
        )

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        resetted = resetted + glorot(self.snn_edge, rng)
        resetted = resetted + glorot(self.snn_node, rng)
        resetted = resetted + self.gnnx2.reset(rng)
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
        edge_embeds: torch.Tensor
        node_embeds: torch.Tensor

        # TODO:
        # Ensure edge features are already dynamic tensors.
        ...

        # In sequence-then-graph flow, dynamic edges are already aggregated
        # together and all steps has exactly the same aggregated edge data, and
        # we will only use data from last step.
        (edge_embeds, _) = self.snn_edge.forward(edge_feats)
        total_edges = total_edges + len(edge_embeds[-1])

        # Take only embedding from the last step.
        # The graph convolution will also use connectivies from last step.
        (node_embeds, _) = (
            self.snn_node.forward(torch.permute(node_feats, (2, 0, 1)))
        )
        # \\ print(edge_feats.shape)
        # \\ print(edge_feats[:, 50].squeeze())

        #
        graph_time_begin = time.time()

        #
        if self.SIMPLEST:
            #
            node_embeds = self.activate(node_embeds[-1])
            node_skips = node_embeds
            edge_embeds = edge_embeds[-1].mean(dim=1, keepdim=True)
            node_embeds = (
                getattr(self.gnnx2.gnn1, "lin").forward(node_embeds)
                + getattr(self.gnnx2.gnn1, "bias")
            )
            res = torch.zeros_like(node_embeds)
            res.index_add_(
                0, edge_tuples[1], node_embeds[edge_tuples[0]] * edge_embeds,
            )
            node_embeds = (
                getattr(self.gnnx2.gnn2, "lin").forward(res)
                + getattr(self.gnnx2.gnn2, "bias")
            )
            res = torch.zeros_like(node_embeds)
            res.index_add_(
                0, edge_tuples[1], node_embeds[edge_tuples[0]] * edge_embeds,
            )
            if isinstance(self.gnnx2, GNNx2Concat):
                #
                node_embeds = torch.cat((res, node_skips), dim=1)
            else:
                #
                node_embeds = res
        else:
            #
            node_embeds = self.gnnx2.forward(
                edge_tuples, edge_embeds[-1], self.activate(node_embeds[-1]),
            )

        #
        graph_time = graph_time + (time.time() - graph_time_begin)
        total_time = time.time() - total_time_begin
        self.COSTS["graph"].append(graph_time)
        self.COSTS["non-graph"].append(total_time - graph_time)
        self.COSTS["edges"].append(total_edges)
        return node_embeds

    def pretrain(self, partname: str, path: str, /) -> None:
        R"""
        Use pretrained model.
        """
        #
        if len(path) == 0:
            #
            return

        #
        if partname == "node":
            #
            reduce = self.reduce_node
            snn = self.snn_node
            print("- \x1b[37mResume pretrained\x1b[0m: \"snn_node\"")
        elif partname == "edge":
            #
            reduce = self.reduce_edge
            snn = self.snn_edge
            print("- \x1b[37mResume pretrained\x1b[0m: \"snn_edge\"")
        else:
            # UNEXPECT:
            # Unknown pretrain part.
            raise NotImplementedError(
                "Unknown pretrain part \"{:s}\".".format(partname),
            )
        pretrain_keys = (
            {
                "gru": (
                    [
                        "weight_ih_l0", "weight_hh_l0", "bias_ih_l0",
                        "bias_hh_l0",
                    ]
                ),
                "lstm": (
                    [
                        "weight_ih_l0", "weight_hh_l0", "bias_ih_l0",
                        "bias_hh_l0",
                    ]
                ),
            }[reduce]
        )

        # Overwrite parameters by pretrained state dict.
        state_dict = torch.load(path)
        state_dict_snn = OrderedDict()
        for key in pretrain_keys:
            #
            state_dict_snn[key] = state_dict["encoder.{:s}".format(key)]
        snn.load_state_dict(state_dict_snn)

        # Freeze overwritten parameters.
        for param in snn.parameters():
            #
            param.requires_grad = False