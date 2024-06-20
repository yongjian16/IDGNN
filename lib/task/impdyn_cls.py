R"""
"""
#
import torch
from typing import Tuple, List, cast
from .dyngraph import DynamicGraph
from ..model.model import Model
from ..model.mlp import MLP
from ..model.activate import activatize
from ..model.initialize import glorot
from .classification import ce_loss, metrics


class ImpDynModelCls(DynamicGraph):
    R"""
    Node windown classification.
    It is a temporal final node classification task.
    """
    def __init__(
        self,
        tgnn: Model, target_num_labels: int, embed_inside_size: int,
        /,
        *,
        label_counts: List[int], activate: str, notembedon: List[int],
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        DynamicGraph.__init__(self)

        #
        self.target_num_labels = target_num_labels
        label_weights = torch.Tensor(label_counts).to(torch.float32)
        label_weights = label_weights / torch.sum(label_weights)
        self.register_buffer("label_weights", label_weights)

        #
        self.tgnn = tgnn

        #
        self.notembedon = notembedon
        if len(self.notembedon) == 0:
            #
            self.mlp = (
                MLP(
                    self.tgnn.feat_target_size, target_num_labels,
                    embed_inside_size,
                    activate=activate,
                )
            )
            self.activate = activatize(activate)
        else:
            # UNEXPECT:
            # Moving average is not supported.
            raise RuntimeError("Moving average not supported.")

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        resetted = resetted + self.tgnn.reset(rng)
        if len(self.notembedon) == 0:
            #
            resetted = resetted + self.mlp.reset(rng)
        return resetted

    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        edge_labels: torch.Tensor, edge_ranges: torch.Tensor,
        edge_times: torch.Tensor, node_feats: torch.Tensor,
        node_labels: torch.Tensor, node_times: torch.Tensor,
        node_masks: torch.Tensor,
        /,
        Z = None,
        A_rho = None,
    ) -> List[torch.Tensor]:
        R"""
        Forward.
        """
        #
        node_embeds: torch.Tensor

        # We do not have label emebdding layers.
        if edge_labels.ndim > 0 or node_labels.ndim > 0:
            # UNEXPECT:
            # Current tasks does not assume any edge or node label input
            # embeddings.
            raise NotImplementedError(
                "Edge or node label input is not supported.",
            )

        # Edge timestamp reshaping is ignored.
        if edge_feats.ndim > 2:
            #
            edge_feats = torch.permute(edge_feats, (2, 0, 1))
        elif edge_feats.ndim == 0:
            #
            edge_feats = (
                torch.ones(
                    (cast(int, torch.max(edge_ranges).item()), 1),
                    dtype=node_feats.dtype, device=edge_tuples.device,
                )
            )

        #
        node_embeds = (
            self.tgnn.forward(
                edge_tuples, edge_feats, edge_ranges, edge_times, node_feats,
                node_times, node_masks, Z=Z, A_rho=A_rho,
            )
        )
        if len(self.notembedon) == 0:
            #
            predictions = self.mlp(self.activate(node_embeds))

        return [node_embeds], predictions
    
    def predict(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        edge_labels: torch.Tensor, edge_ranges: torch.Tensor,
        edge_times: torch.Tensor, node_feats: torch.Tensor,
        node_labels: torch.Tensor, node_times: torch.Tensor,
        node_masks: torch.Tensor,
        /,
        Z = None,
    ) -> List[torch.Tensor]:
        R"""
        Forward.
        """
        #
        node_embeds: torch.Tensor

        # We do not have label emebdding layers.
        if edge_labels.ndim > 0 or node_labels.ndim > 0:
            # UNEXPECT:
            # Current tasks does not assume any edge or node label input
            # embeddings.
            raise NotImplementedError(
                "Edge or node label input is not supported.",
            )

        # Edge timestamp reshaping is ignored.
        if edge_feats.ndim > 2:
            #
            edge_feats = torch.permute(edge_feats, (2, 0, 1))
        elif edge_feats.ndim == 0:
            #
            edge_feats = (
                torch.ones(
                    (cast(int, torch.max(edge_ranges).item()), 1),
                    dtype=node_feats.dtype, device=edge_tuples.device,
                )
            )

        #
        node_embeds = (
            self.tgnn.predict(
                edge_tuples, edge_feats, edge_ranges, edge_times, node_feats,
                node_times, node_masks, Z=Z,
            )
        )
        if len(self.notembedon) == 0:
            #
            predictions = self.mlp(self.activate(node_embeds))
        else:
            #
            predictions = node_embeds
        return predictions
    

    def sidestep(self, /, *ARGS) -> None:
        R"""
        Forward and backward.
        """
        #
        torch.autograd.set_detect_anomaly(True)
        loss, *others = self.loss(*ARGS)
        loss.backward(retain_graph=True)
        return loss, *others

    def loss(self, /, *ARGS) -> torch.Tensor:
        R"""
        Loss funtion.
        """
        #
        node_output_feats: torch.Tensor
        node_target_labels: torch.Tensor
        node_masks: torch.Tensor

        # Output only has node feature-like data.
        # Target node label data are not useful in this task.
        (node_output_feats, _, node_target_labels, node_masks, 
        Z, Z_1, Z_0, V_0, eta_1, eta_2) = ARGS

        # Format output and target data.
        node_exists = node_masks > 0
        node_output_feats = (
            torch.reshape(
                node_output_feats,
                (len(node_output_feats), self.target_num_labels),
            )[node_exists]
        )
        node_target_labels = (
            torch.reshape(node_target_labels, (len(node_target_labels),))
            [node_exists]
        )

        upper_loss = (
            ce_loss(
                node_output_feats, node_target_labels,
                cast(torch.Tensor, self.label_weights),
            )
        )
        device = Z.device

        lower_loss = ((Z_1.reshape(-1)-Z.reshape(-1))**2).sum()
        org_shape = Z_0.shape
        z_grad = torch.autograd.grad(upper_loss, Z_1, retain_graph=True)[0]    

        Z_0 = (1 - eta_1)*Z_0 + eta_1*Z.reshape(*org_shape).detach().cpu()
        g_z_grad = torch.autograd.grad(lower_loss, Z_1, retain_graph=True, create_graph=True)[0]

        hv = torch.sum(g_z_grad*V_0.to(device), dim=-1)

        phi_v = torch.autograd.grad(hv.sum(), Z_1, retain_graph=True)[0]
        V_0 = V_0 - eta_2*phi_v.cpu() + eta_2*z_grad.cpu()

        loss = upper_loss - torch.sum(g_z_grad*V_0.to(device))
        return loss, Z_0, V_0


    def metrics(self, /, *ARGS) -> List[Tuple[int, float]]:
        R"""
        Evaluation metrics.
        """
        #
        node_output_feats: torch.Tensor
        node_target_labels: torch.Tensor
        node_masks: torch.Tensor

        # Output only has node feature-like data.
        # Target node label data are not useful in this task.
        (node_output_feats, _, node_target_labels, node_masks) = ARGS

        # Format output and target data.
        node_exists = node_masks > 0
        node_output_feats = (
            torch.reshape(
                node_output_feats,
                (len(node_output_feats), self.target_num_labels),
            )[node_exists]
        )
        node_target_labels = (
            torch.reshape(node_target_labels, (len(node_target_labels),))
            [node_exists]
        )
        return (
            metrics(
                node_output_feats, node_target_labels,
                cast(torch.Tensor, self.label_weights),
            )
        )