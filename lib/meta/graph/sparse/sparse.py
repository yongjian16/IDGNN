R"""
"""
#
import numpy as onp
import numpy.typing as onpt
from typing import Callable, cast
from ..graph import Graph
from .edge import edge_symmetrize, edge_sort, edge_unique
from .degree import bins
from ....utils.info import INFO
from ...utils.repr.setlike import setlike
from ...types import MEMPIN


class AdjacencyList(Graph):
    R"""
    Graph as adjacency list meta.
    """
    def __init__(
        self,
        edge_srcs: onpt.NDArray[onp.generic],
        edge_dsts: onpt.NDArray[onp.generic],
        edge_feats: onpt.NDArray[onp.generic],
        edge_labels: onpt.NDArray[onp.generic], 
        node_feats: onpt.NDArray[onp.generic],
        node_labels: onpt.NDArray[onp.generic],
        /,
        *,
        hetero: bool, symmetrize: bool, sort: bool,
    ) -> None:
        R"""
        Initialize the class.
        """
        # Safety check.
        if not len(edge_srcs) == len(edge_dsts) == len(edge_feats):
            # EXPECT:
            # It is possible to have dirty data.
            raise RuntimeError(
                "Fail to transform provided data into graph as adjacency list "
                "since its edge defintion columns do not agree on length.",
            )
        if not len(edge_feats) == len(edge_labels):
            # EXPECT:
            # It is possible to have dirty data.
            raise RuntimeError(
                "Fail to transform provided data into graph as adjacency list "
                "since its edge data columns do not agree on length.",
            )
        if not len(node_feats) == len(node_labels):
            # EXPECT:
            # It is possible to have dirty data.
            raise RuntimeError(
                "Fail to transform provided data into graph as adjacency list "
                "since its node data columns do not agree on length.",
            )
        if len(edge_feats.shape) < 2 or len(edge_labels.shape) < 2:
            # EXPECT:
            # It is possible to have dirty data.
            raise RuntimeError(
                "Fail to transform provided data into graph as adjacency list "
                "since its edge data columns need at least 2 dimensions.",
            )
        if len(node_feats.shape) < 2 or len(node_labels.shape) < 2:
            # EXPECT:
            # It is possible to have dirty data.
            raise RuntimeError(
                "Fail to transform provided data into graph as adjacency list "
                "since its node data columns need at least 2 dimensions.",
            )

        # Pay attention that symmetrized graph must be symmetric graph, but
        # symmetric graph may not be symmetrized.
        if symmetrize:
            #
            (edge_srcs, edge_dsts, edge_feats, edge_labels) = (
                edge_symmetrize(edge_srcs, edge_dsts, edge_feats, edge_labels)
            )
            self.symmetrized = True
        else:
            #
            self.symmetrized = False

        #
        if sort:
            #
            (edge_srcs, edge_dsts, edge_feats, edge_labels) = (
                edge_sort(edge_srcs, edge_dsts, edge_feats, edge_labels)
            )
            self.sorted = True
        else:
            #
            self.sorted = False

        # Safety check.
        if (
            not hetero
            and edge_unique(edge_srcs, edge_dsts, sorted=self.sorted)
        ):
            # EXPECT:
            # It is possible to have improper arguments.
            raise RuntimeError(
                "Fail to transform provided data into graph as adjacency list "
                "since non-heterogeneous graph should not have duplicate "
                "edges.",
            )

        #
        meaningless = lambda array: onp.min(array) == onp.max(array)

        # Collect edge data.
        self.edge_srcs_col = 0
        self.edge_dsts_col = 1
        self.edge_tuples = onp.stack((edge_srcs, edge_dsts))
        self.edge_feats = edge_feats
        self.edge_labels = edge_labels

        # Collect node data.
        self.node_feats = node_feats
        self.node_labels = node_labels

        # Collect data shapes.
        self.shapize()

        # Collect meaningless data.
        self.meaningless_node_feats = meaningless(self.node_feats)
        self.meaningless_node_labels = meaningless(self.node_labels)
        self.meaningless_edge_feats = meaningless(self.edge_feats)
        self.meaningless_edge_labels = meaningless(self.edge_labels)

        # Collect essential statistics.
        self.collect()

    @property
    def edge_srcs(self, /) -> onpt.NDArray[onp.generic]:
        R"""
        Get edge source indices.
        """
        # Must explicitly do the annotation.
        return (
            cast(
                onpt.NDArray[onp.generic],
                self.edge_tuples[self.edge_srcs_col],
            )
        )

    @property
    def edge_dsts(self, /) -> onpt.NDArray[onp.generic]:
        R"""
        Get edge destination indices.
        """
        # Must explicitly do the annotation.
        return (
            cast(
                onpt.NDArray[onp.generic],
                self.edge_tuples[self.edge_dsts_col],
            )
        )

    def shapize(self, /) -> None:
        R"""
        Get data shapes.
        """
        #F
        (self.num_edges, self.edge_feat_size) = self.edge_feats.shape
        (_, self.edge_label_size) = self.edge_labels.shape

        #
        (self.num_nodes, self.node_feat_size) = self.node_feats.shape
        (_, self.node_label_size) = self.node_labels.shape

    def collect(self, /) -> None:
        R"""
        Collect essential statistics.
        """
        #
        self.degree_ins = (
            onp.zeros_like(self.edge_tuples, shape=(self.num_nodes,))
        )
        self.degree_outs = (
            onp.zeros_like(self.edge_tuples, shape=(self.num_nodes,))
        )
        onp.add.at(
            self.degree_ins, cast(onpt.NDArray[onp.int64], self.edge_dsts), 1,
        )
        onp.add.at(
            self.degree_outs, cast(onpt.NDArray[onp.int64], self.edge_srcs), 1,
        )

    def repr(self, /) -> INFO:
        R"""
        Get essential statistics for representation.
        """
        # TODO:
        # Replace representation parameters by global or class constants.
        dict = Graph.repr(self)
        dict["(Deg)ree"]["Deg In"] = setlike(self.degree_ins, axis=0, n=1)[0]
        dict["(Deg)ree"]["Deg Out"] = setlike(self.degree_outs, axis=0, n=1)[0]
        dict["(Deg)ree"]["Deg In Dist"] = (
            bins(self.degree_ins, num_bins=10, num_bins_per_line=3)[0]
        )
        dict["(Deg)ree"]["Deg Out Dist"] = (
            bins(self.degree_outs, num_bins=10, num_bins_per_line=3)[0]
        )
        return dict

    def argsort_node_degree_in(self, /) -> onpt.NDArray[onp.generic]:
        R"""
        Argsort nodes by in degrees (large to small).
        """
        #
        return onp.argsort(-self.degree_ins)

    def argsort_edge_degree_in(
        self,
        f: Callable[
            [onpt.NDArray[onp.generic], onpt.NDArray[onp.generic]],
            onpt.NDArray[onp.generic],
        ],
        /,
    ) -> onpt.NDArray[onp.generic]:
        R"""
        Argsort edges by the maxmimum of its source and destination degree
        scores (large to small).
        In most cases, degree score is the maximum of source and destination
        degrees.
        """
        #
        return (
            onp.argsort(
                -f(
                    self.degree_ins[self.edge_srcs],
                    self.degree_ins[self.edge_dsts],
                ),
            )
        )

    def pin(self, batch_size: int, /) -> MEMPIN:
        R"""
        Always-shared data to be pinned into device memory.
        Shared data should not differentiate input and target.
        """
        # Edge info is fixed, so pin it into device memory before the training
        # loop.
        # We need to extend for multi-graph batching.
        # Pay attention that pinning is specific only for same-topology graphs:
        # All graphs in the batch has exactly the same edge connectivities,
        # features and lables.
        # If not, graphs in the batch have no shared memory to be pinned.
        edge_tuples_numpy = self.edge_tuples
        edge_feats_numpy = self.edge_feats
        edge_labels_numpy = self.edge_labels

        # Recreate new adjacency list of the graph batching all graphs
        # together.
        edge_tuples_numpy = (
            onp.reshape(edge_tuples_numpy, (2, 1, self.num_edges))
        )
        edge_feats_numpy = (
            onp.reshape(edge_feats_numpy, (1, *self.edge_feats.shape))
        )
        edge_labels_numpy = (
            onp.reshape(edge_labels_numpy, (1, *self.edge_labels.shape))
        )
        edge_tuples_numpy = onp.tile(edge_tuples_numpy, (1, batch_size, 1))
        edge_feats_numpy = (
            onp.tile(
                edge_feats_numpy,
                (batch_size, *([1] * len(self.edge_feats.shape))),
            )
        )
        edge_labels_numpy = (
            onp.tile(
                edge_labels_numpy,
                (batch_size, *([1] * len(self.edge_labels.shape))),
            )
        )

        # Shift node IDs to batch nodes from different graphs into new graph.
        edge_tuples_numpy = (
            edge_tuples_numpy
            + (
                onp.reshape(
                    onp.arange(batch_size) * self.num_nodes,
                    (1, batch_size, 1),
                )
            )
        )

        # Format edge data.
        edge_tuples_numpy = (
            onp.reshape(edge_tuples_numpy, (2, batch_size * self.num_edges))
        )
        edge_feats_numpy = onp.concatenate(edge_feats_numpy, axis=0)
        edge_labels_numpy = onp.concatenate(edge_labels_numpy, axis=0)

        # Use 0-D vector as meaningless replacement
        if self.meaningless_edge_feats:
            #
            edge_feats_numpy = onp.array(0.0)
        if self.meaningless_edge_labels:
            #
            edge_labels_numpy = onp.array(0)
        return [edge_tuples_numpy, edge_feats_numpy, edge_labels_numpy]