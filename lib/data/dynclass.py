R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import torch
import os
from typing import Optional, List, Dict, Tuple, cast
from ..meta.dyngraph.sparse.dynedge import DynamicAdjacencyListDynamicEdge
import collections
import pickle
from ..model.utils import get_spectral_rad, aug_normalized_adjacency, sparse_mx_to_torch_sparse_tensor
from scipy.sparse import coo_array, coo_matrix

class DynamicClassification(object):
    R"""
    Dynamic node classification over the whole window dataset.
    """
    #
    SOURCE: str

    def __init__(
        self,
        dirname: str,
        /,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        self.from_raw(dirname)
        self.sanitize_edge()

    def from_raw(self, dirname: str, /) -> None:
        R"""
        Load from raw data.
        """
        #
        matrices = (
            onp.load(os.path.join(dirname, "{:s}.npz".format(self.SOURCE)))
        )
        feats = matrices["attmats"] # Brain10 (5000, 12, 20)
        adjmats = matrices["adjs"] # Brain10 (12, 5000, 5000)
        onehots = matrices["labels"] # Brain10 (5000, 10)
        #
        # A_list = [ sparse_mx_to_torch_sparse_tensor(aug_normalized_adjacency(coo_matrix(adj, dtype=float))) for adj in adjmats]
        # # A_list = [ torch.tensor(adj, dtype=torch.float).to(device).to_sparse() for adj in data['adjs']]
        # A_rho = [get_spectral_rad(A) for A in A_list]
        # self.A_rho = A_rho
        # self.A_list = A_list
        #
        # Safety check.
        if onp.any(adjmats < 0):
            # UNEXPECT:
            # Edge weights must be non-negative.
            raise NotImplementedError("Get invalid negative edge weights.")
        if onp.any(onp.sum(onehots > 0, axis=1) != 1):
            # UNEXPECT:
            # Node label must be unique.
            raise NotImplementedError("Get empty or duplicate node labels.")

        #
        self.raw_node_feats = feats
        (_, self.raw_node_labels) = onp.nonzero(onehots)
        (_, self.num_labels) = onehots.shape
        self.label_counts = onp.sum(onehots, axis=0).tolist()

        #
        self.raw_edge_srcs = []
        self.raw_edge_dsts = []
        self.raw_edge_feats = []
        (_, self.num_times, _) = feats.shape
        for t in range(self.num_times):
            #
            (dsts, srcs) = onp.nonzero(adjmats[t]) # Brain10 dsts.shape = srcs.shape = [(154094,), (164190,), ...]
            weights = adjmats[t, dsts, srcs].astype(float) # Brain10 weights.shape = (154094,)
            if onp.any(weights != 1):
                # UNEXPECT:
                # Node label must be unique.
                raise NotImplementedError("Edge has non-0/1 weight.")
            self.raw_edge_srcs.append(srcs)
            self.raw_edge_dsts.append(dsts)
            self.raw_edge_feats.append(weights)
        self.timestamps = onp.arange(self.num_times, dtype=onp.float64)

    def sanitize_edge(self, /) -> None:
        R"""
        Santiize edge data.
        """
        # All dynamic classification tasks are assumed to be directed graphs.
        self.edge_srcs = self.raw_edge_srcs
        self.edge_dsts = self.raw_edge_dsts
        self.edge_feats = self.raw_edge_feats
        self.edge_hetero = False
        self.edge_symmetric = False

    def asto_dynamic_adjacency_list_dynamic_edge(
        self,
        /,
        *,
        window_history_size: Optional[int], window_future_size: int,
        win_aggr: str, timestamped_edge_times: List[str],
        timestamped_node_times: List[str], timestamped_edge_feats: List[str],
        timestamped_node_feats: List[str],
    ) -> DynamicAdjacencyListDynamicEdge:
        R"""
        Transform dataset as temporal adjacency list (static edge) metaset.
        """
        #
        node_feats = onp.transpose(self.raw_node_feats, (0, 2, 1))
        num_nodes = len(node_feats)
        node_labels = onp.reshape(self.raw_node_labels, (num_nodes, 1))

        #
        edge_feats = []
        edge_labels = []
        for t in range(self.num_times):
            #
            num_edges = len(self.edge_feats[t])
            edge_feats.append(onp.reshape(self.edge_feats[t], (num_edges, 1)))
            edge_labels.append(
                cast(
                    onpt.NDArray[onp.generic],
                    onp.zeros((num_edges, 1)).astype(onp.int64),
                ),
            )

        #
        metaset = (
            DynamicAdjacencyListDynamicEdge(
                self.edge_srcs, self.edge_dsts, edge_feats, edge_labels,
                node_feats, node_labels,
                hetero=self.edge_hetero, symmetrize=self.edge_symmetric,
                sort=True,
            )
        )
        metaset.dynamicon(dyn_node_feats=True, dyn_node_labels=False)
        metaset.timestamping(
            self.timestamps,
            timestamped_edge_times=timestamped_edge_times,
            timestamped_node_times=timestamped_node_times,
            timestamped_edge_feats=timestamped_edge_feats,
            timestamped_node_feats=timestamped_node_feats,
        )
        metaset.sliding_window(
            window_history_size=window_history_size,
            window_future_size=window_future_size,
        )
        metaset.sliding_aggregation(win_aggr=win_aggr)
        # customized info
        if hasattr(self, 'lab_avail_nodes'):
            metaset.lab_avail_nodes = self.lab_avail_nodes
        else:
            metaset.lab_avail_nodes = None

        if hasattr(self, 'A_rho'):
            metaset.A_rho = self.A_rho
            metaset.A_list = self.A_list
        return metaset


class Reddit4(DynamicClassification):
    #
    SOURCE = "Reddit4"


class DBLP5(DynamicClassification):
    #
    SOURCE = "DBLP5"


class Brain10(DynamicClassification):
    #
    SOURCE = "Brain10"


class DynCSL(DynamicClassification):
    #
    SOURCE = "DynCSL"

    def from_raw(self, dirname: str, /) -> None:
        R"""
        Load from raw data.
        """
        #
        (data, properties) = (
            torch.load(os.path.join(dirname, "tgnn-power-v2.pt"))
        )
        num_nodes = properties["num_nodes"]
        num_labels = properties["num_labels"]
        num_times = properties["num_timestamps"]
        print(properties)
        
        #
        self.raw_edge_srcs = []
        self.raw_edge_dsts = []
        self.raw_edge_feats = []
        graph_label_buf = []
        for (l, (graph_pair, label)) in enumerate(data):
            #
            (feats, (adjlists, _)) = graph_pair
            for adjlist in adjlists:
                #
                src_snap = []
                dst_snap = []
                for (src, dsts) in enumerate(adjlist):
                    #
                    for dst in dsts:
                        #
                        src_snap.append(src)
                        dst_snap.append(dst)
                if torch.min(feats) != 1 or torch.max(feats) != 1:
                    # UNEXPECT:
                    # DynCSL node features are non-trivial.
                    raise NotImplementedError(
                        "DynCSL node features are non-trivial.",
                    )
                if len(src_snap) != num_nodes * 5:
                    # UNEXPECT:
                    # DynCSL is incomplete.
                    raise NotImplementedError(
                        "Incomplete DynCSL source nodes.",
                    )
                if len(dst_snap) != num_nodes * 5:
                    # UNEXPECT:
                    # DynCSL is incomplete.
                    raise NotImplementedError(
                        "Incomplete DynCSL destination nodes.",
                    )
                self.raw_edge_srcs.append(onp.array(src_snap))
                self.raw_edge_dsts.append(onp.array(dst_snap))
                self.raw_edge_feats.append(onp.ones((num_nodes * 5,)))
                graph_label_buf.append(-1)
            graph_label_buf[-1] = label.item()

            if graph_label_buf[-1] != l % num_labels:
                # UNEXPECT:
                # DynCSL is incomplete.
                raise NotImplementedError(
                    "DynCSL labels should averagely distribute in raw "
                    "sequence.",
                )
        graph_labels = onp.array(graph_label_buf)
        if len(graph_labels) != num_times * len(data):
            # UNEXPECT:
            # DynCSL is incomplete.
            raise NotImplementedError(
                "Some DynCSL has not have {:d} timestamps.".format(num_times),
            )

        #
        self.raw_node_feats = onp.array([[[0.0, 1.0]]])
        self.raw_node_feats = (
            onp.tile(self.raw_node_feats, (num_nodes, len(graph_labels), 1))
        )
        self.raw_node_labels = (
            onp.reshape(graph_labels, (1, len(graph_label_buf), 1))
        )
        self.raw_node_labels = (
            onp.tile(self.raw_node_labels, (num_nodes, 1, 1))
        )

        #
        self.win_size = num_times
        self.num_times = self.win_size * len(data)
        self.num_labels = num_labels
        self.timestamps = onp.arange(self.win_size, dtype=onp.float64)
        self.timestamps = onp.tile(self.timestamps, (len(data),))

    def asto_dynamic_adjacency_list_dynamic_edge(
        self,
        /,
        *,
        window_history_size: Optional[int], window_future_size: int,
        win_aggr: str, timestamped_edge_times: List[str],
        timestamped_node_times: List[str], timestamped_edge_feats: List[str],
        timestamped_node_feats: List[str],
    ) -> DynamicAdjacencyListDynamicEdge:
        R"""
        Transform dataset as temporal adjacency list (static edge) metaset.
        """
        #
        node_feats = onp.transpose(self.raw_node_feats, (0, 2, 1))
        (num_nodes, _, num_times) = node_feats.shape
        node_labels = (
            onp.reshape(self.raw_node_labels, (num_nodes, 1, num_times))
        )

        #
        edge_feats = []
        edge_labels = []
        for t in range(self.num_times):
            #
            num_edges = len(self.edge_feats[t])
            edge_feats.append(onp.reshape(self.edge_feats[t], (num_edges, 1)))
            edge_labels.append(
                cast(
                    onpt.NDArray[onp.generic],
                    onp.zeros((num_edges, 1)).astype(onp.int64),
                ),
            )

        #
        metaset = (
            DynamicAdjacencyListDynamicEdge(
                self.edge_srcs, self.edge_dsts, edge_feats, edge_labels,
                node_feats, node_labels,
                hetero=self.edge_hetero, symmetrize=self.edge_symmetric,
                sort=True,
            )
        )
        metaset.set_win_shift(self.win_size)
        metaset.dynamicon(dyn_node_feats=True, dyn_node_labels=True)
        metaset.timestamping(
            self.timestamps,
            timestamped_edge_times=timestamped_edge_times,
            timestamped_node_times=timestamped_node_times,
            timestamped_edge_feats=timestamped_edge_feats,
            timestamped_node_feats=timestamped_node_feats,
        )
        metaset.sliding_window(
            window_history_size=window_history_size,
            window_future_size=window_future_size,
        )
        metaset.sliding_aggregation(win_aggr=win_aggr)
        return metaset
    
