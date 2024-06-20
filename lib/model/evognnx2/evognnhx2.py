R"""
"""
#
import torch
import torch_geometric as thgeo
import numpy as onp
import time
from typing import cast
from .evognnx2 import EvoGNNx2


class TopKSummary(torch.nn.Module):
    R"""
    Top-k summarization module fpr EvolveGCN-H.
    """
    def __init__(self, num_feats: int, num_selects: int, /):
        R"""
        Initialize the class.
        """
        #
        torch.nn.Module.__init__(self)

        # This is a vector used to do dot product with each feature vector to
        # get the score for top-k feature vector selection.
        self.weight = torch.nn.parameter.Parameter(torch.zeros(num_feats, 1))
        self.num_selects = num_selects

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        # Initialize the vector as linear bias, but based on input rather than
        # output.
        (fan_in, _) = self.weight.shape
        a = onp.sqrt(6 / (fan_in + fan_in))
        self.weight.data.uniform_(-a, a, generator=rng)
        return self.weight.numel()

    def forward(
        self,
        feats: torch.Tensor, masks: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        # Compute feature vector scores and select only top-k vectors with
        # padding for robustness.
        # Explicitly and dirtily cast the padding index to pass static typing.
        scores = torch.mm(feats, self.weight) / torch.norm(self.weight)
        scores = scores.squeeze()
        masks = masks.squeeze()
        scores[masks <= 0] = float("-inf")
        (topk_scores, topk_indices) = (
            torch.topk(scores, min(len(scores), self.num_selects))
        )
        topk_indices = topk_indices[topk_scores > float("-inf")]
        topk_pads = (
            torch.full(
                (self.num_selects - len(topk_indices),),
                cast(int, topk_indices[-1]),
                dtype=topk_indices.dtype, device=topk_indices.device,
            )
        )

        # Select feature vectors and re-weigt by scores.
        topk_indices = torch.cat((topk_indices, topk_pads))
        selects = (
            feats[topk_indices]
            * (
                torch.reshape(
                    torch.tanh(scores[topk_indices]), (self.num_selects, 1),
                )
            )
        )
        return selects


class EvoGNNHx2(EvoGNNx2):
    R"""
    EvolveGCN-H Graph neural network (2-layer).
    """
    #
    EVOBY = "H"
    REDUCE = "gru"
    RNN = "GRU"

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
        EvoGNNx2.__init__(
            self, feat_input_size_edge, feat_input_size_node, feat_target_size,
            embed_inside_size,
            convolve=convolve, reduce=reduce, skip=skip, activate=activate,
        )

        #
        self.summary1 = TopKSummary(feat_input_size_node, embed_inside_size)
        self.summary2 = TopKSummary(embed_inside_size, feat_target_size)

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = EvoGNNx2.reset(self, rng)
        resetted = resetted + self.summary1.reset(rng)
        resetted = resetted + self.summary2.reset(rng)
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
        # EngCovid
        # edge_tuples: torch.Size([2, 36042])
        # edge_feats: torch.Size([36042, 1])
        # edge_ranges: torch.Size([2, 7])
        # tensor([[    0,  5524, 10652, 15470, 20875, 26153, 31341],
        #         [ 5524, 10652, 15470, 20875, 26153, 31341, 36042]])

        # edge_times: torch.Size([]), tensor(0.)
        # node_feats: torch.Size([516, 1, 7]) # 516 nodes, 1 feature, 7 time steps
        # node_times: torch.Size([]), tensor(0.)
        # node_masks: torch.Size([516])

        total_time_begin = time.time()
        graph_time = 0.0
        total_edges = 0

        #
        node_embeds: torch.Tensor

        #
        (_, num_times) = edge_ranges.shape

        # Pay attention that weight matrices in EvolveGCN are parameters. while
        # weight attributes of GCN are just place holders.
        # They are different, and should be treated differently.
        weights = [self.weight1, self.weight2]
        summaries = [self.summary1, self.summary2]
        rnns = [self.rnn1, self.rnn2]
        gnns = [self.gnnx2.gnn1, self.gnnx2.gnn2]
        activates = [self.activate1, self.activate2]

        #
        for t in range(num_times):
            # Get edge slice for current snapshot.
            # Must explicitly use built-in slice function to pass static
            # typing.
            edge_snap_slice = slice(edge_ranges[0, t], edge_ranges[1, t])

            #
            edge_snap_tuples = edge_tuples[:, edge_snap_slice]
            edge_snap_feats = edge_feats[edge_snap_slice]
            edge_snap_embeds = (
                self.edge_activate(
                    self.edge_transform.forward(edge_snap_feats),
                )
            )
            node_embeds = node_feats[:, :, t]

            #
            for l in range(2):
                # Evolve weight matrix.
                weights[l] = (
                    rnns[l].forward(
                        summaries[l].forward(node_embeds, node_masks),
                        weights[l],
                    )
                )

                # Overwrite GCN weight by latest evolution and convolve by it.
                # Pay attention that the weight attribute here is a place
                # holder to accept weight as input but still follow PyTorch
                # Geometric design, and it is not learnable parameter.
                graph_time_begin = time.time()
                lin = cast(thgeo.nn.dense.linear.Linear, gnns[l].lin)
                lin.weight = weights[l]
                node_embeds = (
                    gnns[l].forward(
                        activates[l](node_embeds), edge_snap_tuples,
                        edge_snap_embeds.squeeze(),
                    )
                )
                graph_time = graph_time + (time.time() - graph_time_begin)
            total_edges = total_edges + edge_snap_tuples.shape[1]
        total_time = time.time() - total_time_begin
        self.COSTS["graph"].append(graph_time)
        self.COSTS["non-graph"].append(total_time - graph_time)
        self.COSTS["edges"].append(total_edges)
        return node_embeds