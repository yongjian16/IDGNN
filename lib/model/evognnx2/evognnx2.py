R"""
"""
#
import abc
import torch
import torch_geometric as thgeo
from typing import List, cast
from ..model import Model
from ..gnnx2 import GNNx2 as GNNx2Skip
from ..activate import activatize
from ..snn import sequentialize
from ..initialize import glorot


class GNNx2(GNNx2Skip):
    R"""
    Graph neural network (2-layer) without any skip-connection.
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

        # This is just used as a holder for GNNs in EvolveGCN.
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

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        resetted = resetted + glorot(self.gnn1, rng)
        resetted = resetted + glorot(self.gnn2, rng)
        return resetted

    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        # UNEXPECT:
        # We need to explicitly use each GNN layer one-by-one in EvolveGCN.
        raise NotImplementedError(
            "EvolveGCN can not run all graph convolutions together as an atom."
        )


class EvoGNNx2(Model):
    R"""
    EvolveGCN Graph neural network (2-layer).
    """
    #
    EVOBY: str
    REDUCE: str
    RNN: str

    def __init__(
        self,
        feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int, embed_inside_size: int,
        /,
        *,
        convolve: str, reduce: str, skip: bool, activate: str,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        # Safety check.
        if convolve != "gcnub":
            # UNEXPECT:
            # EvolveGCN only supports GCN without bias.
            raise NotImplementedError(
                "EvolveGCN only supports GCN without bias.",
            )
        if reduce != self.REDUCE:
            # UNEXPECT:
            # EvolveGCN subclasses require specific sequential reducer.
            raise NotImplementedError(
                "EvolveGCN-{:s} requires {:s} sequential reducer."
                .format(self.EVOBY, self.RNN),
            )
        if skip:
            # UNEXPECT:
            # EvolveGCN does not support skip-connection.
            raise NotImplementedError(
                "EvolveGCN does not support skip-connection.",
            )

        #
        self.gnnx2 = (
            GNNx2(
                feat_input_size_edge, feat_input_size_node,
                feat_target_size, embed_inside_size,
                convolve=convolve, skip=skip, activate=activate,
            )
        )

        #
        self.edge_transform: torch.nn.Module

        #
        if feat_input_size_edge > 1 and convolve in ("gcn", "gcnub", "cheb"):
            #
            self.edge_transform = torch.nn.Linear(feat_input_size_edge, 1)
            self.edge_activate = activatize("softplus")
        else:
            self.edge_transform = torch.nn.Identity()
            self.edge_activate = activatize("identity")

        #
        self.activate1 = activatize("identity")
        self.activate2 = activatize(activate)

        # Sequence models here will taking weight matrix as input, and directly
        # do matrix multiplication.
        # EvolveGCN only supports recurrent neural networks.
        self.rnn1 = (
            sequentialize(
                reduce + "[]", feat_input_size_node, feat_input_size_node,
            )
        )
        self.rnn2 = (
            sequentialize(reduce + "[]", embed_inside_size, embed_inside_size)
        )

        #
        self.activate = activatize(activate)

        # Swap weight matrix to current level.
        lin1 = cast(thgeo.nn.dense.linear.Linear, self.gnnx2.gnn1.lin)
        lin2 = cast(thgeo.nn.dense.linear.Linear, self.gnnx2.gnn2.lin)
        self.weight1 = lin1.weight
        self.weight2 = lin2.weight
        del lin1.weight
        del lin2.weight
        lin1.weight = self.weight1.data
        lin2.weight = self.weight2.data

        #
        self.feat_target_size = feat_target_size

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        # Enfore evolving weight to be initialized by 0.
        self.weight1.data.zero_()
        self.weight2.data.zero_()

        #
        resetted = 0
        resetted = resetted + self.weight1.numel() + self.weight2.numel()
        resetted = resetted + glorot(self.edge_transform, rng)
        resetted = resetted + glorot(self.rnn1, rng)
        resetted = resetted + glorot(self.rnn2, rng)
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
        #
        ...