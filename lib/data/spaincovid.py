R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import os
import h5py
from typing import Optional, List, Dict, Tuple
from ..meta.dyngraph.sparse.staedge import DynamicAdjacencyListStaticEdge


class SpainCOVID(object):
    R"""
    Spain COVID-19 dataset.
    """
    # The multiplex graph is determined by vehicle types.
    VEHICLES = ["boat", "bus", "car", "plane", "train"]

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
        comfix = "weighted-multiplex/data"
        dataset = (
            h5py.File(os.path.join(dirname, "spain-covid19-dataset.h5"), "r")
        )

        #
        prefix = "{:s}/networks/d0".format(comfix)
        networks = [
            (
                onp.array(dataset["{:s}/{:s}/edge_list".format(prefix, key)]),
                onp.array(
                    dataset["{:s}/{:s}/edge_attr/weight".format(prefix, key)],
                ),
                onp.array(dataset["{:s}/{:s}/node_list".format(prefix, key)]),
                onp.array(
                    dataset
                    ["{:s}/{:s}/node_attr/population".format(prefix, key)],
                ),
            )
            for key in self.VEHICLES
        ]
        population = max(pop for (_, _, _, pop) in networks)
        self.raw_edge_srcs = (
            onp.concatenate(
                [edge_tuples[:, 0] for (edge_tuples, _, _, _) in networks],
                axis=0,
            )
        )
        self.raw_edge_dsts = (
            onp.concatenate(
                [edge_tuples[:, 1] for (edge_tuples, _, _, _) in networks],
                axis=0,
            )
        )
        self.raw_edge_labels = (
            (
                onp.concatenate(
                    [
                        onp.full((len(edge_feats),), i)
                        for i, (_, edge_feats, _, _) in enumerate(networks)
                    ],
                )
            )
        )
        self.raw_edge_feats = (
            onp.concatenate([edge_feats for (_, edge_feats, _, _) in networks])
            * population
        )
        self.raw_nodes = (
            onp.array(dataset["weighted-multiplex/data/timeseries/d0"])
        )

        # Timestamp is day.
        self.timestamps = onp.arange(len(self.raw_nodes), dtype=onp.float64)
        import pdb;pdb.set_trace()
        
    def sanitize_edge(self, /) -> None:
        R"""
        Santiize edge data.
        """
        # Spain COVID-19 will performs better as directed graph.
        self.edge_srcs = self.raw_edge_srcs
        self.edge_dsts = self.raw_edge_dsts
        self.edge_labels = self.raw_edge_labels 
        self.edge_feats = self.raw_edge_feats
        self.edge_hetero = True
        self.edge_symmetric = False

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
        node_feats = (
            onp.reshape(self.raw_nodes, (*self.raw_nodes.shape, 1))
        )
        node_feats = onp.transpose(node_feats, (1, 2, 0))
        num_edges = len(self.edge_feats)
        num_nodes = len(node_feats)
        edge_feats = onp.reshape(self.edge_feats, (num_edges, 1))
        edge_labels = (
            onp.reshape(self.edge_labels, (num_edges, 1)).astype(onp.int64)
        )
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