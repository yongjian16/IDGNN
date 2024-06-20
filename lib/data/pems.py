R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import os
import pandas as pd
from typing import Optional, List, Dict, Tuple
from ..meta.dyngraph.sparse.staedge import DynamicAdjacencyListStaticEdge


class PeMS(object):
    R"""
    PeMS dataset.
    """
    #
    DISTRICT: str

    def __init__(
        self,
        dirname: str,
        /,
        *,
        aug_minutes: bool, aug_weekdays: bool,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        self.from_raw(dirname)
        self.sanitize_edge()

        #
        self.raw_nodes: onpt.NDArray[onp.generic]

        # Augment global features by exact timestamps.
        # Gap between different steps are 5 minutes, and we use hour as
        # timestamp unit.
        (num_timestamps, num_nodes, _) = self.raw_nodes.shape
        self.timestamps = onp.arange(num_timestamps) * 5.0 / 24.0

        # Augment node features by minutes.
        # Gap between different steps are 5 minutes.
        if aug_minutes:
            #
            (num_timestamps, num_nodes, _) = self.raw_nodes.shape
            num_day_minutes = 60 // 5 * 24
            num_days = (
                int(onp.ceil(float(num_timestamps) / float(num_day_minutes)))
            )
            day_minutes = onp.arange(num_day_minutes) * 5
            minutes = onp.tile(day_minutes, num_days)[:num_timestamps]
            minutes = minutes.astype(self.raw_nodes.dtype)
            minutes = onp.reshape(minutes, (num_timestamps, 1, 1))
            minutes = onp.tile(minutes, (1, num_nodes, 1))
            self.raw_nodes = onp.concatenate([self.raw_nodes, minutes], 2)

        # Augment node features by weekdays.
        # Gap between different steps are 5 minutes.
        if aug_weekdays:
            #
            (num_timestamps, num_nodes, _) = self.raw_nodes.shape
            num_day_minutes = 60 // 5 * 24
            num_week_minutes = num_day_minutes * 7
            num_weeks = (
                int(onp.ceil(float(num_timestamps) / float(num_week_minutes)))
            )
            weekdays = onp.repeat(onp.arange(7), num_day_minutes)
            weekdays = onp.tile(weekdays, num_weeks)[:num_timestamps]
            weekdays = weekdays.astype(self.raw_nodes.dtype)
            weekdays = onp.reshape(weekdays, (num_timestamps, 1, 1))
            weekdays = onp.tile(weekdays, (1, num_nodes, 1))
            self.raw_nodes = onp.concatenate([self.raw_nodes, weekdays], 2)

    def from_raw(self, dirname: str, /) -> None:
        R"""
        Load from raw data.
        """
        #
        file_edges = "distance.csv"
        file_nodes = "pems{:s}.npz".format(self.DISTRICT)
        raw_edges = pd.read_csv(os.path.join(dirname, file_edges))
        self.raw_edge_srcs = raw_edges["from"].to_numpy()
        self.raw_edge_dsts = raw_edges["to"].to_numpy()
        self.raw_edge_feats = raw_edges["cost"].to_numpy()
        self.raw_nodes = onp.load(os.path.join(dirname, file_nodes))["data"]

    def sanitize_edge(self, /) -> None:
        R"""
        Santiize edge data.
        """
        #
        collects: Dict[Tuple[int, int], List[float]]

        # Remove dirty duplications.
        # Duplications are same undirected connections regardless of edge
        # weights (edge weights should be the same for those duplications).
        collects = {}
        for (src, dst, feat) in (
            zip(self.raw_edge_srcs, self.raw_edge_dsts, self.raw_edge_feats)
        ):
            key = (src.item(), dst.item())
            key = (min(key), max(key))
            val = feat.item()
            if key in collects:
                collects[key].append(val)
            else:
                collects[key] = [val]
        edge_srcs_buf = []
        edge_dsts_buf = []
        edge_feats_buf = []
        for ((src, dst), feats) in collects.items():
            #
            edge_srcs_buf.append(src)
            edge_dsts_buf.append(dst)
            edge_feats_buf.append(sum(feats) / len(feats))
            if min(feats) != max(feats):
                # UNEXPECT:
                # Duplicate edges have different edge features.
                raise NotImplementedError(
                    "PeMS duplicate edges have different edge features.",
                )
        self.edge_srcs = onp.array(edge_srcs_buf)
        self.edge_dsts = onp.array(edge_dsts_buf)
        self.edge_feats = onp.array(edge_feats_buf)
        self.edge_hetero = False
        self.edge_symmetric = True

        #
        if not onp.all(self.edge_feats > 0):
            # UNEXPECT:
            # Edge features as weights must be positive.
            raise NotImplementedError(
                "PeMS{:s} edge weights is not all-positive."
                .format(self.DISTRICT),
            )

    def asto_dynamic_adjacency_list_static_edge(
        self,
        /,
        *,
        window_history_size: Optional[int], window_future_size: int,
        timestamped_edge_times: List[str], timestamped_node_times: List[str],
        timestamped_edge_feats: List[str], timestamped_node_feats: List[str],
    ) -> DynamicAdjacencyListStaticEdge:
        R"""
        Transform dataset as temporal adjacency list (static edge) metaset.
        """
        #
        node_feats = onp.transpose(self.raw_nodes, (1, 2, 0))
        num_edges = len(self.edge_feats)
        num_nodes = len(node_feats)
        edge_feats = onp.reshape(self.edge_feats, (num_edges, 1))
        edge_labels = onp.zeros((num_edges, 1)).astype(onp.int64)
        node_labels = onp.zeros((num_nodes, 1)).astype(onp.int64)
        metaset = (
            DynamicAdjacencyListStaticEdge(
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
        return metaset


class PeMS04(PeMS):
    R"""
    PeMS (district 4) dataset.
    """
    #
    DISTRICT = "04"


class PeMS08(PeMS):
    R"""
    PeMS (district 8) dataset.
    """
    #
    DISTRICT = "08"