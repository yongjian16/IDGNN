R"""
"""
#
import torch
import torch_geometric as thgeo
from typing import cast
from .model import Model
from .activate import activatize
from .initialize import glorot
from .snn import auto_num_heads


class GNNx2(Model):
    R"""
    Graph neural network (2-layer).
    """
    def __init__(
        self,
        feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int, embed_inside_size: int,
        /,
        *,
        convolve: str, skip: bool, activate: str,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        # TODO:
        # Given more than 2 layers, we can introduce dense connection.
        self.gnn1 = (
            self.graphicalize(
                convolve, feat_input_size_edge, feat_input_size_node,
                embed_inside_size,
                activate=activate,
            )
        )
        self.gnn2 = (
            self.graphicalize(
                convolve, feat_input_size_edge, embed_inside_size,
                feat_target_size,
                activate=activate,
            )
        )

        #
        self.edge_transform: torch.nn.Module
        self.skip: torch.nn.Module

        #
        if feat_input_size_edge > 1 and convolve in ("gcn", "gcnub", "cheb"):
            #
            self.edge_transform = torch.nn.Linear(feat_input_size_edge, 1)
            self.edge_activate = activatize("softplus")
        else:
            self.edge_transform = torch.nn.Identity()
            self.edge_activate = activatize("identity")

        #
        if feat_input_size_node == feat_target_size:
            #
            self.skip = torch.nn.Identity()
        else:
            #
            self.skip = (
                torch.nn.Linear(feat_input_size_node, feat_target_size)
            )

        #
        self.activate = activatize(activate)

        # Use a 0-or-1 integer to mask skip connection.
        self.doskip = int(skip)

    def graphicalize(
        self,
        name: str, feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int,
        /,
        *,
        activate: str,
    ) -> torch.nn.Module:
        R"""
        Get unit graphical module.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        if name == "gcn":
            #
            module = (
                thgeo.nn.GCNConv(feat_input_size_node, feat_target_size)
            )
        elif name == "gcnub":
            #
            module = (
                thgeo.nn.GCNConv(
                    feat_input_size_node, feat_target_size,
                    bias=False,
                )
            )
        elif name == "gat":
            #
            heads = auto_num_heads(feat_target_size)
            module = (
                thgeo.nn.GATConv(
                    feat_input_size_node, feat_target_size // heads,
                    heads=heads, edge_dim=feat_input_size_edge,
                )
            )
        elif name == "cheb":
            #
            module = (
                thgeo.nn.ChebConv(feat_input_size_node, feat_target_size, 2)
            )
        elif name == "gin":
            #
            module = (
                thgeo.nn.GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            feat_input_size_node, feat_target_size,
                        ),
                        activatize(activate),
                        torch.nn.Linear(feat_target_size, feat_target_size),
                    ),
                    edge_dim=feat_input_size_edge,
                )
            )
        else:
            # EXPECT:
            # It is possible to require unsupporting sequential model.
            raise RuntimeError(
                "Graphical module identifier \"{:s}\" is not supported."
                .format(name),
            )
        return cast(torch.nn.Module, module)

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        resetted = resetted + glorot(self.gnn1, rng)
        resetted = resetted + glorot(self.gnn2, rng)
        resetted = resetted + glorot(self.edge_transform, rng)
        resetted = resetted + glorot(self.skip, rng)
        return resetted

    def convolve(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Convolve.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        node_embeds: torch.Tensor

        #
        node_embeds = (
            self.gnn1.forward(node_feats, edge_tuples, edge_feats.squeeze())
        )
        node_embeds = (
            self.gnn2.forward(
                self.activate(node_embeds), edge_tuples, edge_feats.squeeze(),
            )
        )
        return node_embeds

    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        edge_embeds: torch.Tensor
        node_embeds: torch.Tensor
        node_residuals: torch.Tensor

        #
        edge_embeds = (
            self.edge_activate(self.edge_transform.forward(edge_feats))
        )
        # \\ print(edge_embeds[50])
        node_embeds = self.convolve(edge_tuples, edge_embeds, node_feats)
        # \\ print(node_embeds[10, 6].item())
        node_residuals = self.skip.forward(node_feats)
        return node_embeds + self.doskip * node_residuals


class GNNx2Concat(GNNx2):
    R"""
    Graph neural network (2-layer) with input concatenation.
    """
    #
    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        node_embeds: torch.Tensor

        # Super call.
        node_embeds = GNNx2.forward(self, edge_tuples, edge_feats, node_feats)
        node_embeds = torch.cat((node_embeds, node_feats), dim=1)
        return node_embeds


def graphicalize(
    name: str, feat_input_size_edge, feat_input_size_node: int,
    feat_target_size: int, embed_inside_size: int,
    /,
    *,
    skip: bool, activate: str, concat: bool,
) -> Model:
    R"""
    Get 2-layer graphical module.
    """
    #
    if concat:
        #
        return (
            GNNx2Concat(
                feat_input_size_edge, feat_input_size_node, feat_target_size,
                embed_inside_size,
                convolve=name, skip=skip, activate=activate,
            )
        )
    else:
        #
        return (
            GNNx2(
                feat_input_size_edge, feat_input_size_node, feat_target_size,
                embed_inside_size,
                convolve=name, skip=skip, activate=activate,
            )
        )