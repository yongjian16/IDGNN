R"""
"""
#
import torch
import torch_geometric as thgeo
import time
from typing import List, cast
from .evognnx2 import EvoGNNx2


class EvoGNNOx2(EvoGNNx2):
    R"""
    EvolveGCN-O Graph neural network (2-layer).
    """
    #
    EVOBY = "O"
    REDUCE = "lstm"
    RNN = "LSTM"

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

        #
        (_, num_times) = edge_ranges.shape

        # Pay attention that weight matrices in EvolveGCN are parameters. while
        # weight attributes of GCN are just place holders.
        # They are different, and should be treated differently.
        weights = [self.weight1, self.weight2]
        transients = (
            [
                torch.zeros_like(self.weight1.data),
                torch.zeros_like(self.weight2.data),
            ]
        )
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
                (weights[l], transients[l]) = (
                    rnns[l].forward(weights[l], (weights[l], transients[l]))
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