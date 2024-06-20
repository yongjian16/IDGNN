R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import os
import json
from typing import Optional, List, Dict, Tuple, cast
from ..meta.dyngraph.sparse.dynedge import DynamicAdjacencyListDynamicEdge


class EngCOVID(object):
    R"""
    England COVID-19 dataset.
    """
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
        with open(os.path.join(dirname, "england_covid.json"), "r") as file:
            #
            dataset = json.load(file)

        #
        self.num_times = dataset["time_periods"]
        self.raw_edge_srcs = []
        self.raw_edge_dsts = []
        self.raw_edge_feats = []
        for t in range(self.num_times):
            #
            edge_snap_tuple = (
                onp.array(dataset["edge_mapping"]["edge_index"][str(t)])
            )
            edge_snap_srcs = edge_snap_tuple[:, 0]
            edge_snap_dsts = edge_snap_tuple[:, 1]
            edge_snap_feats= (
                onp.array(dataset["edge_mapping"]["edge_weight"][str(t)])
            )
            self.raw_edge_srcs.append(edge_snap_srcs)
            self.raw_edge_dsts.append(edge_snap_dsts)
            self.raw_edge_feats.append(edge_snap_feats)
        self.raw_nodes = onp.array(dataset["y"])
        self.timestamps = onp.arange(self.num_times, dtype=onp.float64)

    def sanitize_edge(self, /) -> None:
        R"""
        Santiize edge data.
        """
        # England COVID-19 will performs better as directed graph.
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
        node_feats = (
            onp.transpose(
                onp.reshape(self.raw_nodes, (*self.raw_nodes.shape, 1)),
                (1, 2, 0),
            )
        )
        num_nodes = len(node_feats)
        node_labels = onp.zeros((num_nodes, 1)).astype(onp.int64)

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
        return metaset