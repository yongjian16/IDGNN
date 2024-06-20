R"""
"""
#
import torch
from typing import Tuple, List, cast
from .nodewinclass import NodeWindowClassification
from ..model.model import Model
from ..model.mlp import MLP
from .classification import ce_loss, metrics


class GraphWindowClassification(NodeWindowClassification):
    R"""
    Graph windown classification.
    It is a temporal final graph classification task.
    """
    #
    SEE_EMBEDS = False

    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        edge_labels: torch.Tensor, edge_ranges: torch.Tensor,
        edge_times: torch.Tensor, node_feats: torch.Tensor,
        node_labels: torch.Tensor, node_times: torch.Tensor,
        node_masks: torch.Tensor,
        /,
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
            num_times = len(edge_feats)
        elif edge_feats.ndim == 0:
            #
            edge_feats = (
                torch.ones(
                    (cast(int, torch.max(edge_ranges).item()), 1),
                    dtype=node_feats.dtype, device=edge_tuples.device,
                )
            )
            (_, num_times) = edge_ranges.shape

        #
        node_feats = (
            torch.ones(
                (len(node_feats), 1, num_times),
                dtype=node_feats.dtype, device=node_feats.device,
            )
        )

        #
        node_embeds = (
            self.tgnn.forward(
                edge_tuples, edge_feats, edge_ranges, edge_times, node_feats,
                node_times, node_masks,
            )
        )
        # \\ if self.SEE_EMBEDS:
        # \\     #
        # \\     print(node_embeds[0, 0].item(), node_embeds[-1, 0].item())
        node_embeds = node_embeds.reshape(len(node_feats) // 19, 19, -1)
        graph_embeds = torch.mean(node_embeds, dim=1)
        if len(self.notembedon) == 0:
            #
            graph_embeds = self.mlp(self.activate(graph_embeds))
        return [graph_embeds]

    def loss(self, /, *ARGS) -> torch.Tensor:
        R"""
        Loss funtion.
        """
        #
        graph_output_feats: torch.Tensor
        node_target_labels: torch.Tensor
        graph_target_labels: torch.Tensor

        # Output only has node feature-like data.
        # Target node label data are not useful in this task.
        (graph_output_feats, _, node_target_labels, _) = ARGS

        graph_target_labels = (
            torch.reshape(
                node_target_labels.squeeze(), (len(graph_output_feats), 19),
            )[:, 0]
        )
        return (
            ce_loss(
                graph_output_feats, graph_target_labels,
                cast(torch.Tensor, self.label_weights),
            )
        )

    def metrics(self, /, *ARGS) -> List[Tuple[int, float]]:
        R"""
        Evaluation metrics.
        """
        #
        graph_output_feats: torch.Tensor
        node_target_labels: torch.Tensor
        graph_target_labels: torch.Tensor

        # Output only has node feature-like data.
        # Target node label data are not useful in this task.
        (graph_output_feats, _, node_target_labels, _) = ARGS
        graph_target_labels = (
            torch.reshape(
                node_target_labels.squeeze(), (len(graph_output_feats), 19),
            )[:, 0]
        )
        return (
            metrics(
                graph_output_feats, graph_target_labels,
                cast(torch.Tensor, self.label_weights),
            )
        )