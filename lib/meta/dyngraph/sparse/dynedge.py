R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import scipy.sparse as osparse
import os
import torch
from typing import List, Callable, Tuple, cast, Optional, Union
from ..dyngraph import DynamicGraph
from ...graph.sparse.edge import edge_symmetrize, edge_sort, edge_unique
from ...graph.sparse.degree import bins
from ....utils.info import INFO, info5, infounion
from ...utils.repr.setlike import setlike
from ...types import MEMPIN, MEMBAT
from ...utils.normalize import normalize, standardize
from ...utils.repr.distrep import distrep
from ...utils.spindle import fitsplit
from ...utils.slidewin import sliding_window
from ...vecseq import VectorSequence
import time
from tqdm.auto import tqdm
# from p_tqdm import p_map, p_umap, p_imap, p_uimap

class DynamicAdjacencyListDynamicEdge(DynamicGraph):
    R"""
    Temporal graph as dynamic adjacency list meta whose edge connectivities,
    features and labels are also dynamic.
    """
    #
    WINAGG_NONE = 0
    WINAGG_DENSE = 1
    WINAGG_SPARSE = 2
    WINAGGS = {
        "none": WINAGG_NONE,
        "dense": WINAGG_DENSE,
        "sparse": WINAGG_SPARSE,
    }

    #
    WIN_SHIFT = -1

    def set_win_shift(self, num: int, /,) -> None:
        R"""
        Change window shifting factor.
        """
        # Should only support DynCSL.
        if num != 8:
            # UNEXPECT:
            # Only support DynCSL.
            raise NotImplementedError(
                "Only support non-default window shifting 8 for DynCSL.",
            )
        self.WIN_SHIFT = num

    def __init__(
        self,
        edge_srcs: List[onpt.NDArray[onp.generic]],
        edge_dsts: List[onpt.NDArray[onp.generic]],
        edge_feats: List[onpt.NDArray[onp.generic]],
        edge_labels: List[onpt.NDArray[onp.generic]],
        node_feats: onpt.NDArray[onp.generic],
        node_labels: onpt.NDArray[onp.generic],
        /,
        *,
        hetero: bool, symmetrize: bool, sort: bool,
    ) -> None:
        R"""
        Initialize the class.
        """
        # Safety check at each snapshot.
        for (
            t,
            (
                edge_snap_srcs, edge_snap_dsts, edge_snap_feats,
                edge_snap_labels,
            ),
        ) in enumerate(zip(edge_srcs, edge_dsts, edge_feats, edge_labels)):
            #
            if (
                not (
                    len(edge_snap_srcs) == len(edge_snap_dsts)
                    == len(edge_snap_feats)
                )
            ):
                # EXPECT:
                # It is possible to have dirty data.
                raise RuntimeError(
                    "Fail to transform provided data of timestamp {:d} into "
                    "graph as adjacency list since its edge defintion columns "
                    "do not agree on length.".format(t),
                )
            if not len(edge_snap_feats) == len(edge_snap_labels):
                # EXPECT:
                # It is possible to have dirty data.
                raise RuntimeError(
                    "Fail to transform provided data of timestamp {:d} into "
                    "graph as adjacency list since its edge data columns do "
                    "not agree on length.".format(t),
                )
            if (
                len(edge_snap_feats.shape) < 2
                or len(edge_snap_labels.shape) < 2
            ):
                # EXPECT:
                # It is possible to have dirty data.
                raise RuntimeError(
                    "Fail to transform provided data of timestamp {:d} into "
                    "graph as adjacency list since its edge data columns need "
                    "at least 2 dimensions.".format(t),
                )

        # Safety check over all nodes of all timestamps.
        if not len(node_feats) == len(node_labels):
            # EXPECT:
            # It is possible to have dirty data.
            raise RuntimeError(
                "Fail to transform provided data into graph as dynamic "
                "adjacency list since its node data columns do not agree on "
                "length.",
            )
        if len(node_feats.shape) < 2 or len(node_labels.shape) < 2:
            # EXPECT:
            # It is possible to have dirty data.
            raise RuntimeError(
                "Fail to transform provided data into graph as dynamic "
                "adjacency list since its node data columns need at least 2 "
                "dimensions.",
            )

        # Pay attention that symmetrized graph must be symmetric graph, but
        # symmetric graph may not be symmetrized.
        if symmetrize:
            #
            edge_srcs_new = []
            edge_dsts_new = []
            edge_feats_new = []
            edge_labels_new = []
            for (
                edge_snap_srcs, edge_snap_dsts, edge_snap_feats,
                edge_snap_labels,
            ) in zip(edge_srcs, edge_dsts, edge_feats, edge_labels):
                #
                (
                    edge_snap_srcs, edge_snap_dsts, edge_snap_feats,
                    edge_snap_labels,
                ) = (
                    edge_symmetrize(
                        edge_snap_srcs, edge_snap_dsts, edge_snap_feats,
                        edge_snap_labels,
                    )
                )
                edge_srcs_new.append(edge_snap_srcs)
                edge_dsts_new.append(edge_snap_dsts)
                edge_feats_new.append(edge_snap_feats)
                edge_labels_new.append(edge_snap_labels)
            edge_srcs = edge_srcs_new
            edge_dsts = edge_dsts_new
            edge_feats = edge_feats_new
            edge_labels = edge_labels_new
            self.symmetrized = True
        else:
            #
            self.symmetrized = False

        #
        if sort:
            #
            #
            edge_srcs_new = []
            edge_dsts_new = []
            edge_feats_new = []
            edge_labels_new = []
            for (
                edge_snap_srcs, edge_snap_dsts, edge_snap_feats,
                edge_snap_labels,
            ) in zip(edge_srcs, edge_dsts, edge_feats, edge_labels):
                #
                (
                    edge_snap_srcs, edge_snap_dsts, edge_snap_feats,
                    edge_snap_labels,
                ) = (
                    edge_sort(
                        edge_snap_srcs, edge_snap_dsts, edge_snap_feats,
                        edge_snap_labels,
                    )
                )
                edge_srcs_new.append(edge_snap_srcs)
                edge_dsts_new.append(edge_snap_dsts)
                edge_feats_new.append(edge_snap_feats)
                edge_labels_new.append(edge_snap_labels)
            edge_srcs = edge_srcs_new
            edge_dsts = edge_dsts_new
            edge_feats = edge_feats_new
            edge_labels = edge_labels_new
            self.sorted = True
        else:
            #
            self.sorted = False

        # Safety check.
        if not hetero:
            #
            for (t, (edge_snap_srcs, edge_snap_dsts)) in (
                enumerate(zip(edge_srcs, edge_dsts))
            ):
                #
                if (
                    edge_unique(
                        edge_snap_srcs, edge_snap_dsts,
                        sorted=self.sorted,
                    )
                ):
                    # EXPECT:
                    # It is possible to have improper arguments.
                    raise RuntimeError(
                        "Fail to transform provided data into graph as "
                        "dynamic adjacency list since non-heterogeneous graph "
                        "should not have duplicate edges at timestamp {:d}."
                        .format(t),
                    )

        #
        meaninglessall = lambda array: onp.min(array) == onp.max(array)
        meaninglesssnap = (
            lambda lstarray: (
                any(onp.min(array) == onp.max(array) for array in lstarray)
            )
        )

        # Collect edge data.
        self.edge_srcs_col = 0
        self.edge_dsts_col = 1
        self.edge_tuples = (
            [
                onp.stack((edge_snap_srcs, edge_snap_dsts))
                for (edge_snap_srcs, edge_snap_dsts) in (
                    zip(edge_srcs, edge_dsts)
                )
            ]
        )
        self.edge_feats = edge_feats
        self.edge_labels = edge_labels

        # Collect node data.
        self.node_feats = node_feats
        self.node_labels = node_labels

        # Collect data shapes.
        self.shapize()

        # Collect meaningless data.
        self.meaningless_node_feats = meaninglessall(self.node_feats)
        self.meaningless_node_labels = meaninglessall(self.node_labels)
        self.meaningless_edge_feats = meaninglesssnap(self.edge_feats)
        self.meaningless_edge_labels = meaninglesssnap(self.edge_labels)

        # Collect essential statistics.
        self.collect()

    @property
    def edge_srcs(self, /) -> List[onpt.NDArray[onp.generic]]:
        R"""
        Get edge source indices.
        """
        # Must explicitly do the annotation.
        return (
            [
                cast(
                    onpt.NDArray[onp.generic],
                    edge_snap_tuples[self.edge_srcs_col],
                )
                for edge_snap_tuples in self.edge_tuples
            ]
        )

    @property
    def edge_dsts(self, /) -> List[onpt.NDArray[onp.generic]]:
        R"""
        Get edge destination indices.
        """
        # Must explicitly do the annotation.
        return (
            [
                cast(
                    onpt.NDArray[onp.generic],
                    edge_snap_tuples[self.edge_dsts_col],
                )
                for edge_snap_tuples in self.edge_tuples
            ]
        )

    def shapize(self, /) -> None:
        R"""
        Get data shapes.
        """
        # Edge data must be dynamic as a list of arraies.
        num_edges_buf = []
        for (edge_snap_feats, edge_snap_labels) in (
            zip(self.edge_feats, self.edge_labels)
        ):
            #
            (num_edges, self.edge_feat_size) = edge_snap_feats.shape
            (_, self.edge_label_size) = edge_snap_labels.shape
            num_edges_buf.append(num_edges)
        self.num_edges_array = onp.array(num_edges_buf)
        self.num_edges = onp.sum(self.num_edges_array).item()

        # Node data may have time axis which will requires explicitly setting
        # sliding window.
        (self.num_nodes, *self.node_feat_shape) = self.node_feats.shape
        (_, *self.node_label_shape) = self.node_labels.shape

    def collect(self, /) -> None:
        R"""
        Collect essential statistics.
        """
        #
        degree_ins_buf = []
        degree_outs_buf = []
        for (edge_snap_srcs, edge_snap_dsts) in (
            zip(self.edge_srcs, self.edge_dsts)
        ):
            #
            degree_ins = (
                onp.zeros_like(self.num_edges, shape=(self.num_nodes,))
            )
            degree_outs = (
                onp.zeros_like(self.num_edges, shape=(self.num_nodes,))
            )
            onp.add.at(
                degree_ins, cast(onpt.NDArray[onp.int64], edge_snap_dsts), 1,
            )
            onp.add.at(
                degree_outs, cast(onpt.NDArray[onp.int64], edge_snap_srcs), 1,
            )
            degree_ins_buf.append(degree_ins)
            degree_outs_buf.append(degree_outs)
        self.degree_ins = onp.stack(degree_ins_buf, axis=1)
        self.degree_outs = onp.stack(degree_outs_buf, axis=1)

    def dynamicon(
        self,
        /,
        *,
        dyn_node_feats: bool, dyn_node_labels: bool,
    ) -> None:
        R"""
        Specify dynamic data columns.
        """
        #
        if not (dyn_node_feats or dyn_node_labels):
            # EXPECT:
            # It is possible to have improper arguments.
            raise RuntimeError(
                "Fail to transform provided data into temporal graph as "
                "dynamic adjacency list with static edges since none of node"
                "columns is temporal.",
            )

        # Update node shapes.
        if dyn_node_feats:
            #
            (self.node_feat_size, self.num_times) = self.node_feat_shape
        else:
            #
            (self.node_feat_size,) = self.node_feat_shape
        self.dyn_node_feats = dyn_node_feats
        if dyn_node_labels:
            #
            (self.node_label_size, self.num_times) = self.node_label_shape
        else:
            #
            (self.node_label_size,) = self.node_label_shape
        self.dyn_node_labels = dyn_node_labels

    def timestamping(
        self,
        timestamps: onpt.NDArray[onp.generic],
        /,
        *,
        timestamped_edge_times: List[str], timestamped_node_times: List[str],
        timestamped_edge_feats: List[str], timestamped_node_feats: List[str],
    ) -> None:
        R"""
        Set exact timestamps for dynamic graph steps.
        """
        #
        DynamicGraph.timestamping(
            self, timestamps,
            timestamped_edge_times=timestamped_edge_times,
            timestamped_edge_feats=timestamped_edge_feats,
            timestamped_node_times=timestamped_node_times,
            timestamped_node_feats=timestamped_node_feats,
        )

        # Edge timestamps.
        self.edge_times_abs = (
            [
                onp.full((edge_snap_tuples.shape[1],), snap_t)
                for (snap_t, edge_snap_tuples) in (
                    zip(self.timestamps.tolist(), self.edge_tuples)
                )
            ]
        )
        self.edge_times_inc = (
            [onp.zeros_like(self.edge_times_abs[0])]
            + (
                [
                    onp.full((edge_snap_tuples.shape[1],), curr_t - prev_t)
                    for (prev_t, curr_t, edge_snap_tuples) in (
                        zip(
                            self.timestamps[:-1].tolist(),
                            self.timestamps[1:].tolist(), self.edge_tuples[1:],
                        )
                    )
                ]
            )
        )

        # Node timestamps.
        self.node_times_abs = (
            onp.tile(
                onp.reshape(self.timestamps, (1, self.num_times)),
                (self.num_nodes, 1),
            )
        )
        self.node_times_inc = (
            onp.concatenate(
                (
                    onp.zeros_like(
                        self.node_times_abs,
                        shape=(self.num_nodes, 1),
                    ),
                    self.node_times_abs[:, 1:] - self.node_times_abs[:, :-1],
                ),
                axis=1,
            )
        )

    def sliding_window(
        self,
        /,
        *,
        window_history_size: Optional[int], window_future_size: Optional[int],
    ) -> None:
        R"""
        Extend dynamic graph with sliding window.
        """
        # Get number of windows.
        (
            self.window_size, self.window_history_size,
            self.window_future_size,
        ) = (
            sliding_window(
                window_history_size, window_future_size, self.num_times,
            )
        )
        if self.WIN_SHIFT < 0:
            #
            self.num_windows = self.num_times - self.window_size + 1
        else:
            # DynCSL.
            self.num_windows = self.num_times // self.WIN_SHIFT

    def sliding_aggregation(self, *, win_aggr: str) -> None:
        R"""
        Set sliding window aggregation method.
        """
        #
        self.winaggr = self.WINAGGS[win_aggr]

    def repr(self, /) -> INFO:
        R"""
        Get essential statistics for representation.
        """
        #
        dict = DynamicGraph.repr(self)
        dict["Basic"]["#Edges"] = (
            "[{:d}, {:d}]({:.1f})".format(
                onp.min(self.num_edges_array), onp.max(self.num_edges_array),
                onp.mean(self.num_edges_array),
            )
        )
        dict["Basic"]["Win.Aggr"] = ["None", "Dense", "Sparse"][self.winaggr]

        # By default, we focus on the degree sum over all timestamps.
        dict["(Deg)ree Max"] = {}
        dict["(Deg)ree Sum"] = {}
        dict["(Deg)ree"] = dict["(Deg)ree Sum"]

        #
        dict["(Deg)ree Max"]["Deg In"] = (
            setlike(onp.max(self.degree_ins, axis=1), axis=0, n=1)[0]
        )
        dict["(Deg)ree Max"]["Deg Out"] = (
            setlike(onp.max(self.degree_outs, axis=1), axis=0, n=1)[0]
        )
        dict["(Deg)ree Max"]["Deg In Dist"] = (
            bins(
                onp.max(self.degree_ins, axis=1),
                num_bins=10, num_bins_per_line=3,
            )[0]
        )
        dict["(Deg)ree Max"]["Deg Out Dist"] = (
            bins(
                onp.max(self.degree_outs, axis=1),
                num_bins=10, num_bins_per_line=3,
            )[0]
        )

        #
        dict["(Deg)ree Sum"]["Deg In"] = (
            setlike(onp.sum(self.degree_ins, axis=1), axis=0, n=1)[0]
        )
        dict["(Deg)ree Sum"]["Deg Out"] = (
            setlike(onp.sum(self.degree_outs, axis=1), axis=0, n=1)[0]
        )
        dict["(Deg)ree Sum"]["Deg In Dist"] = (
            bins(
                onp.sum(self.degree_ins, axis=1),
                num_bins=10, num_bins_per_line=3,
            )[0]
        )
        dict["(Deg)ree Sum"]["Deg Out Dist"] = (
            bins(
                onp.sum(self.degree_outs, axis=1),
                num_bins=10, num_bins_per_line=3,
            )[0]
        )
        return dict

    def argsort_node_degree_in(self, /) -> onpt.NDArray[onp.generic]:
        R"""
        Argsort nodes by in degrees (large to small).
        """
        #
        return onp.argsort(-onp.sum(self.degree_ins, axis=1))

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

    def fitsplit(
        self,
        proportion: Tuple[int, int, int], priority: Tuple[int, int, int],
        spindle: str,
        /,
    ) -> Tuple[
        onpt.NDArray[onp.generic], onpt.NDArray[onp.generic],
        onpt.NDArray[onp.generic],
    ]:
        R"""
        Split data indices into training, validation and test indices along
        dimension identifier by given name.
        """
        # Node split should ensure degree balance.
        # Time split just do the directly cut on spindle axis.
        if hasattr(self, "lab_avail_nodes"):
            lab_avail_nodes = self.lab_avail_nodes
        else:
            lab_avail_nodes = None

        (indices_train, indices_valid, indices_test) = (
            fitsplit(
                {
                    "node": self.argsort_node_degree_in(),
                    "time": onp.arange(self.num_windows),
                }[spindle],
                list(proportion), list(priority), 
                lab_avail_nodes if spindle == 'node' else None,
            )
        )
        if self.WIN_SHIFT > 0:
            #
            (train_size, valid_size, test_size) = proportion
            scale = self.num_windows // sum(proportion)
            train_size = train_size * scale
            valid_size = valid_size * scale + train_size
            test_size = test_size * scale + valid_size
            indices_train = onp.arange(0, train_size)
            indices_valid = onp.arange(train_size, valid_size)
            indices_test = onp.arange(valid_size, test_size)
        return (indices_train, indices_valid, indices_test)

    def normalizeby(
        self,
        indices: onpt.NDArray[onp.generic], spindle: str,
        /,
    ) -> List[List[Tuple[float, float]]]:
        R"""
        Normalize data using statistics from indices.
        """
        # We may need to additionally normalize on time dimense for edge
        # features.
        if spindle == "node":
            #
            fromarray_buf = self.edge_feats # Brain10 [(154094, 1), (164190, 1), ...]
        elif spindle == "time":
            #
            fromarray_buf = [self.edge_feats[i] for i in indices.tolist()]
        else:
            # UNEXPECT:
            # Unknown normalization.
            raise NotImplementedError(
                "Unkonw normalization spindle \"{:s}\".".format(spindle),
            )
        fromarray = onp.concatenate(fromarray_buf, axis=0)
        edge_feats_new = []

        for edge_snap_feats in self.edge_feats:
            #
            (edge_snap_feats, edge_factors) = (
                normalize(edge_snap_feats, fromarray, axis=0)
            )
            edge_feats_new.append(edge_snap_feats)
        if not self.meaningless_edge_feats:
            #
            self.edge_feats = edge_feats_new

        # We may need to additionally normalize on time dimense for node
        # features.
        vdim = {"node": 0, "time": 2}[spindle]
        axis = (0, 2) if self.dyn_node_feats else (0,)

        (self.node_feats, node_factors) = (
            normalize(
                self.node_feats, # Brain10 (5000, 20, 12)
                onp.take(
                    self.node_feats, cast(onpt.NDArray[onp.int64], indices),
                    vdim,
                ),
                axis=axis,
            )
        )
        return [edge_factors, node_factors]

    def distrep(self, /, *, n: int) -> str:
        R"""
        Get data distribution representation of the class.
        """
        #
        axis = lambda flag: (0, 2) if flag else (0,)
        fade = (
            lambda string, flag: (
                "\n".join(
                    "\x1b[90m" + line + "\x1b[0m"
                    for line in string.split("\n")
                )
                if flag else
                string
            )
        )

        #
        return (
            info5(
                {
                    "Data (Dist)ribute": {
                        "Edge Feat": (
                            fade(
                                distrep(
                                    onp.concatenate(self.edge_feats, axis=0),
                                    axis=0, n=n,
                                )[0],
                                self.meaningless_edge_feats,
                            )
                        ),
                        "Edge Label": (
                            fade(
                                distrep(
                                    onp.concatenate(self.edge_labels, axis=0),
                                    axis=0, n=n,
                                )[0],
                                self.meaningless_edge_labels,
                            )
                        ),
                        "Node Feat": (
                            fade(
                                distrep(
                                    self.node_feats,
                                    axis=axis(self.dyn_node_feats), n=n,
                                )[0],
                                self.meaningless_node_feats,
                            )
                        ),
                        "Node Label": (
                            fade(
                                distrep(
                                    self.node_labels,
                                    axis=axis(self.dyn_node_labels), n=n,
                                )[0],
                                self.meaningless_node_labels,
                            )
                        ),
                    }
                },
            )
        )

    def pin(self, batch_size: int, /) -> MEMPIN:
        R"""
        Always-shared data to be pinned into device memory.
        Shared data should not differentiate input and target.
        """
        # Corner case optimization.
        if self.winaggr == self.WINAGG_NONE:
            #
            return []
        if (
            len(self) == 1 and self.node_feat_size == 20
            and self.num_times == 12
        ):
            #
            corner_case = "Brain10"
        elif (
            len(self) == 1 and self.node_feat_size == 100
            and self.num_times == 10
        ):
            #
            corner_case = "DBLP5"
        elif (
            len(self) == 1 and self.node_feat_size == 20
            and self.num_times == 10
        ):
            #
            corner_case = "Reddit4"
        else:
            #
            self.cached = []
            return []
        if self.winaggr != self.WINAGG_DENSE:
            # EXPECT:
            # Open to be finisied.
            raise RuntimeError("Unimplemented corner case.")

        #
        path = "{:s}.cache".format(corner_case)
        if os.path.isfile(path):
            #
            self.cached = torch.load(path)
            return []

        # Get input and target window ties.
        input_tie1 = 0
        input_tie2 = input_tie1 + self.window_history_size
        target_tie1 = input_tie2
        target_tie2 = target_tie1 + self.window_future_size
        input_begin = min(input_tie1, input_tie2)
        input_end = max(input_tie1, input_tie2)
        target_begin = min(target_tie1, target_tie2)
        target_end = max(target_tie1, target_tie2)

        # Get edge timestamp data of current batch.
        edge_times_abs_input = self.edge_times_abs[input_begin:input_end]
        edge_times_rel_input = (
            [
                onp.full_like(
                    edge_times_abs_it,
                    (
                        onp.max(edge_times_abs_input[-1])
                        - onp.min(edge_times_abs_it)
                    ),
                )
                for edge_times_abs_it in edge_times_abs_input
            ]
        )
        edge_times_inc_input = self.edge_times_inc[input_begin:input_end]

        # Get node timestamp data of current batch.
        node_times_abs_input = self.node_times_abs[:, input_begin:input_end]
        node_times_rel_input = (
            onp.reshape(node_times_abs_input[:, -1], (self.num_nodes, 1))
            - node_times_abs_input
        )
        node_times_inc_input = self.node_times_inc[:, input_begin:input_end]

        # Collect unique edge identifiers as integers.
        node_max = (
            max(
                onp.max(edge_tuples_it).item()
                for edge_tuples_it in (
                    self.edge_tuples[input_begin:input_end]
                )
            )
        )
        edge_ints = (
            [
                edge_srcs_it * (node_max + 1) + edge_dsts_it
                for (edge_srcs_it, edge_dsts_it) in (
                    zip(
                        self.edge_srcs[input_begin:input_end],
                        self.edge_dsts[input_begin:input_end],
                    )
                )
            ]
        )
        edge_uniqints = onp.sort(onp.unique(onp.concatenate(edge_ints)))

        # Safety check.
        if self.edge_feat_size > 1:
            # UNEXPECT:
            # Only scalar temporal edge feature is aggregatable.
            raise NotImplementedError(
                "Only scalar temporal edge feature is aggregatable.",
            )
        if (
            any(
                onp.any(onp.less(edge_feats_it, 0.0))
                for edge_feats_it in self.edge_feats
            )
        ):
            # UNEXPECT:
            # Padding value can not exist in raw edge feature.
            raise NotImplementedError(
                "negative values are used as padding values in dense "
                "feature aggregation, thus can not be part of edge "
                "feature.",
            )

        # Generate dense edge feature and appearance aggregation as sparse
        # matrices.
        aggr_eids = onp.concatenate(edge_ints)
        aggr_tids = (
            onp.concatenate(
                [
                    onp.full_like(edge_ints_it, t)
                    for (t, edge_ints_it) in enumerate(edge_ints)
                ],
            )
        )
        aggr_vals_feat = (
            onp.reshape(
                onp.concatenate(self.edge_feats[input_begin:input_end]),
                (len(aggr_eids),),
            )
        )
        aggr_vals_appr = onp.ones((len(aggr_eids)), dtype=bool)
        aggr_spmat_feat = (
            osparse.coo_matrix(
                (aggr_vals_feat, (aggr_eids, aggr_tids)),
                shape=(
                    onp.max(aggr_eids).item() + 1,
                    self.window_history_size,
                ),
            ).tocsc()
        )
        aggr_spmat_appr = (
            osparse.coo_matrix(
                (aggr_vals_appr, (aggr_eids, aggr_tids)),
                shape=(
                    onp.max(aggr_eids).item() + 1,
                    self.window_history_size,
                ),
            ).tocsc()
        )

        #
        aggr_srcs = edge_uniqints // (node_max + 1)
        aggr_dsts = edge_uniqints % (node_max + 1)
        aggr_tuples = onp.stack((aggr_srcs, aggr_dsts))

        # Convert to dense feature matrix of non-trivial feature dynamics.
        # Use -1.0 as default padding value.
        aggr_feats_buf = []
        aggr_apprs_buf = []
        print('converting to dense feature...')
        tic = time.time()
        for eid in tqdm(edge_uniqints.tolist()):
            #
            aggr_feats_buf.append(aggr_spmat_feat[eid].toarray())
            aggr_apprs_buf.append(aggr_spmat_appr[eid].toarray())
            
        # def densify(eid):
        #     return (aggr_spmat_feat[eid].toarray(), aggr_spmat_appr[eid].toarray())
        
        # results = p_map(densify, edge_uniqints.tolist(), **{"num_cpus": 5})
        # aggr_feats_buf = [arri[0] for arri in results]
        # aggr_apprs_buf = [arri[1] for arri in results]

        aggr_feats = (
            onp.reshape(
                onp.stack(aggr_feats_buf),
                (len(edge_uniqints), 1, self.window_history_size),
            )
        )
        aggr_apprs = (
            onp.reshape(
                onp.stack(aggr_apprs_buf),
                (len(edge_uniqints), 1, self.window_history_size),
            )
        )
        aggr_feats[onp.logical_not(aggr_apprs)] = -1.0
        toc = time.time()
        print(f'converting time: {(toc - tic)/60} mins')
        # We should not have any edge labels.
        if not self.meaningless_edge_labels:
            # UNEXPECT
            # We can not aggregate temporal edge labels for now.
            raise NotImplementedError(
                "We can not aggregate temporal edge labels for now.",
            )
        aggr_labels = onp.array(0)

        # Aggregated timestamps will not care the raw appearance since it
        # will regard all edges appearing once as existing all the time.
        aggr_times_abs = (
            onp.concatenate(
                [
                    onp.unique(time_abs)
                    for time_abs in edge_times_abs_input
                ],
            )
        )
        aggr_times_rel = onp.max(aggr_times_abs) - aggr_times_abs
        aggr_times_inc = (
            onp.concatenate(
                (
                    onp.zeros((1,), dtype=aggr_times_abs.dtype),
                    aggr_times_abs[1:] - aggr_times_abs[:-1],
                ),
            )
        )
        aggr_times_abs = onp.tile(aggr_times_abs, (len(edge_uniqints), 1))
        aggr_times_rel = onp.tile(aggr_times_rel, (len(edge_uniqints), 1))
        aggr_times_inc = onp.tile(aggr_times_inc, (len(edge_uniqints), 1))

        # Rename as new edge data.
        edge_tuples_numpy = aggr_tuples
        edge_feats_numpy = aggr_feats
        edge_labels_numpy = aggr_labels
        edge_ranges_numpy = onp.array([0, len(edge_feats_numpy)])

        # Collect edge timestamp data.
        edge_times_numpy_buf = []
        if self.ABS in self.timestamped_edge_times:
            #
            edge_times_numpy_buf.append(aggr_times_abs)
        if self.REL in self.timestamped_edge_times:
            #
            edge_times_numpy_buf.append(aggr_times_rel)
        if self.INC in self.timestamped_edge_times:
            #
            edge_times_numpy_buf.append(aggr_times_inc)
        if len(edge_times_numpy_buf) == 0:
            # Use number of timestamps as place holder.
            edge_times_numpy = onp.array(self.num_times)
        else:
            # Merge all timestamp data into a single array.
            # Put node axis at first for batching.
            edge_times_numpy = onp.stack(edge_times_numpy_buf)
            edge_times_numpy = onp.transpose(edge_times_numpy, (1, 0, 2))

        # A corner case.
        self.cached = (
            [
                edge_tuples_numpy, edge_feats_numpy, edge_labels_numpy,
                edge_ranges_numpy, edge_times_numpy,
            ]
        )
        torch.save(self.cached, path)
        return []

    def __len__(self, /) -> int:
        R"""
        Length of the class.
        """
        #
        return self.num_windows

    def idx_to_timesteps(self, idx: int):
        input_tie1 = idx if self.WIN_SHIFT < 0 else idx * self.WIN_SHIFT
        input_tie2 = input_tie1 + self.window_history_size
        target_tie1 = input_tie2
        target_tie2 = (
            target_tie1 + self.window_future_size
            if self.WIN_SHIFT < 0 else
            target_tie1 - 1
        )

        input_begin = min(input_tie1, input_tie2)
        input_end = max(input_tie1, input_tie2)
        target_begin = min(target_tie1, target_tie2)
        target_end = max(target_tie1, target_tie2)
        return (input_begin, input_end, target_begin, target_end)
    

    def __getitem__(self, idx: int, /) -> MEMBAT:
        R"""
        Get an indexable item of the class.
        """
        # Get input and target window ties.
        # DynCSL is special.
        input_tie1 = idx if self.WIN_SHIFT < 0 else idx * self.WIN_SHIFT
        input_tie2 = input_tie1 + self.window_history_size
        target_tie1 = input_tie2
        target_tie2 = (
            target_tie1 + self.window_future_size
            if self.WIN_SHIFT < 0 else
            target_tie1 - 1
        )

        input_begin = min(input_tie1, input_tie2)
        input_end = max(input_tie1, input_tie2)
        target_begin = min(target_tie1, target_tie2)
        target_end = max(target_tie1, target_tie2)

        # Get edge timestamp data of current batch.
        edge_times_abs_input = self.edge_times_abs[input_begin:input_end]
        edge_times_rel_input = (
            [
                onp.full_like(
                    edge_times_abs_it,
                    (
                        onp.max(edge_times_abs_input[-1])
                        - onp.min(edge_times_abs_it)
                    ),
                )
                for edge_times_abs_it in edge_times_abs_input
            ]
        )
        edge_times_inc_input = self.edge_times_inc[input_begin:input_end]

        # Get node timestamp data of current batch.
        node_times_abs_input = self.node_times_abs[:, input_begin:input_end]
        node_times_rel_input = (
            onp.reshape(node_times_abs_input[:, -1], (self.num_nodes, 1))
            - node_times_abs_input
        )
        node_times_inc_input = self.node_times_inc[:, input_begin:input_end]

        #
        if self.winaggr == self.WINAGG_DENSE and len(self) > 1:
            # Collect unique edge identifiers as integers.
            node_max = (
                max(
                    onp.max(edge_tuples_it).item()
                    for edge_tuples_it in (
                        self.edge_tuples[input_begin:input_end]
                    )
                )
            )
            edge_ints = (
                [
                    edge_srcs_it * (node_max + 1) + edge_dsts_it
                    for (edge_srcs_it, edge_dsts_it) in (
                        zip(
                            self.edge_srcs[input_begin:input_end],
                            self.edge_dsts[input_begin:input_end],
                        )
                    )
                ]
            )
            edge_uniqints = onp.sort(onp.unique(onp.concatenate(edge_ints)))

            # Safety check.
            if self.edge_feat_size > 1:
                # UNEXPECT:
                # Only scalar temporal edge feature is aggregatable.
                raise NotImplementedError(
                    "Only scalar temporal edge feature is aggregatable.",
                )
            if (
                any(
                    onp.any(onp.less(edge_feats_it, 0.0))
                    for edge_feats_it in self.edge_feats
                )
            ):
                # UNEXPECT:
                # Padding value can not exist in raw edge feature.
                raise NotImplementedError(
                    "negative values are used as padding values in dense "
                    "feature aggregation, thus can not be part of edge "
                    "feature.",
                )

            # Generate dense edge feature and appearance aggregation as sparse
            # matrices.
            aggr_eids = onp.concatenate(edge_ints)
            aggr_tids = (
                onp.concatenate(
                    [
                        onp.full_like(edge_ints_it, t)
                        for (t, edge_ints_it) in enumerate(edge_ints)
                    ],
                )
            )
            aggr_vals_feat = (
                onp.reshape(
                    onp.concatenate(self.edge_feats[input_begin:input_end]),
                    (len(aggr_eids),),
                )
            )
            aggr_vals_appr = onp.ones((len(aggr_eids)), dtype=bool)
            aggr_spmat_feat = (
                osparse.coo_matrix(
                    (aggr_vals_feat, (aggr_eids, aggr_tids)),
                    shape=(
                        onp.max(aggr_eids).item() + 1,
                        self.window_history_size,
                    ),
                ).tocsc()
            )
            aggr_spmat_appr = (
                osparse.coo_matrix(
                    (aggr_vals_appr, (aggr_eids, aggr_tids)),
                    shape=(
                        onp.max(aggr_eids).item() + 1,
                        self.window_history_size,
                    ),
                ).tocsc()
            )

            #
            aggr_srcs = edge_uniqints // (node_max + 1)
            aggr_dsts = edge_uniqints % (node_max + 1)
            aggr_tuples = onp.stack((aggr_srcs, aggr_dsts))

            # Convert to dense feature matrix of non-trivial feature dynamics.
            # Use -1.0 as default padding value.
            aggr_feats_buf = []
            aggr_apprs_buf = []
            for eid in edge_uniqints.tolist():
                #
                aggr_feats_buf.append(aggr_spmat_feat[eid].toarray())
                aggr_apprs_buf.append(aggr_spmat_appr[eid].toarray())
            aggr_feats = (
                onp.reshape(
                    onp.stack(aggr_feats_buf),
                    (len(edge_uniqints), 1, self.window_history_size),
                )
            )
            aggr_apprs = (
                onp.reshape(
                    onp.stack(aggr_apprs_buf),
                    (len(edge_uniqints), 1, self.window_history_size),
                )
            )
            aggr_feats[onp.logical_not(aggr_apprs)] = -1.0

            # We should not have any edge labels.
            if not self.meaningless_edge_labels:
                # UNEXPECT
                # We can not aggregate temporal edge labels for now.
                raise NotImplementedError(
                    "We can not aggregate temporal edge labels for now.",
                )
            aggr_labels = onp.array(0)

            # Aggregated timestamps will not care the raw appearance since it
            # will regard all edges appearing once as existing all the time.
            aggr_times_abs = (
                onp.concatenate(
                    [
                        onp.unique(time_abs)
                        for time_abs in edge_times_abs_input
                    ],
                )
            )
            aggr_times_rel = onp.max(aggr_times_abs) - aggr_times_abs
            aggr_times_inc = (
                onp.concatenate(
                    (
                        onp.zeros((1,), dtype=aggr_times_abs.dtype),
                        aggr_times_abs[1:] - aggr_times_abs[:-1],
                    ),
                )
            )
            aggr_times_abs = onp.tile(aggr_times_abs, (len(edge_uniqints), 1))
            aggr_times_rel = onp.tile(aggr_times_rel, (len(edge_uniqints), 1))
            aggr_times_inc = onp.tile(aggr_times_inc, (len(edge_uniqints), 1))

            # Rename as new edge data.
            edge_tuples_numpy = aggr_tuples
            edge_feats_numpy = aggr_feats
            edge_labels_numpy = aggr_labels
            edge_ranges_numpy = onp.array([0, len(edge_feats_numpy)])

            # Collect edge timestamp data.
            edge_times_numpy_buf = []
            if self.ABS in self.timestamped_edge_times:
                #
                edge_times_numpy_buf.append(aggr_times_abs)
            if self.REL in self.timestamped_edge_times:
                #
                edge_times_numpy_buf.append(aggr_times_rel)
            if self.INC in self.timestamped_edge_times:
                #
                edge_times_numpy_buf.append(aggr_times_inc)
            if len(edge_times_numpy_buf) == 0:
                # Use number of timestamps as place holder.
                edge_times_numpy = onp.array(self.num_times)
            else:
                # Merge all timestamp data into a single array.
                # Put node axis at first for batching.
                edge_times_numpy = onp.stack(edge_times_numpy_buf)
                edge_times_numpy = onp.transpose(edge_times_numpy, (1, 0, 2))
        elif self.winaggr == self.WINAGG_NONE:
            # Dynamic graph with dynamic edge has different snapshots with
            # consecutive ranges.
            edge_ranges_numpy = (
                onp.array(
                    [0] + [len(time_abs) for time_abs in edge_times_abs_input],
                )
            )
            edge_ranges_numpy = onp.cumsum(edge_ranges_numpy)
            edge_ranges_numpy = (
                onp.stack((edge_ranges_numpy[:-1], edge_ranges_numpy[1:]))
            )

            # Collect edge timestamp data.
            edge_times_numpy_buf = []
            if self.ABS in self.timestamped_edge_times:
                #
                edge_times_numpy_buf.append(
                    onp.concatenate(edge_times_abs_input),
                )
            if self.REL in self.timestamped_edge_times:
                #
                edge_times_numpy_buf.append(
                    onp.concatenate(edge_times_rel_input),
                )
            if self.INC in self.timestamped_edge_times:
                #
                edge_times_numpy_buf.append(
                    onp.concatenate(edge_times_inc_input),
                )
            if len(edge_times_numpy_buf) == 0:
                # Use number of timestamps as place holder.
                edge_times_numpy = onp.array(self.num_times)
            else:
                # Merge all timestamp data into a single array.
                edge_times_numpy = onp.stack(edge_times_numpy_buf)

            # Collect edge feature and label.
            edge_tuples_numpy = (
                onp.concatenate(
                    self.edge_tuples[input_begin:input_end],
                    axis=1,
                )
            )
            if not self.meaningless_edge_feats:
                #
                edge_feats_numpy = (
                    onp.concatenate(self.edge_feats[input_begin:input_end])
                )
            else:
                #
                edge_feats_numpy = onp.array(0.0)
            if not self.meaningless_edge_labels:
                #
                edge_labels_numpy = (
                    onp.concatenate(self.edge_labels[input_begin:input_end])
                )
            else:
                #
                edge_labels_numpy = onp.array(0)
        elif len(self) > 1:
            # UNEXPECT:
            # Unknown aggregation for given length.
            raise NotImplementedError("Unknown aggregation for given length.")
        # \\ print(edge_tuples_numpy.shape)
        # \\ print(edge_feats_numpy.shape)
        # \\ print(edge_labels_numpy.shape)
        # \\ print(edge_ranges_numpy)
        # \\ print(edge_times_numpy.shape)
        # \\ raise RuntimeError

        def fetch(
            array: onpt.NDArray[onp.generic], dataon: List[int], dynamic: bool,
            tmin: int, tmax: int, null: Union[int, float],
        ) -> onpt.NDArray[onp.generic]:
            R"""
            Fetch online memory from data.
            """
            if dynamic:
                # if dynamic: array.shape = (num_nodes, feat_dim, num_times)
                return (
                    onp.array(null)
                    if len(dataon) == 0 else
                    cast(
                        onpt.NDArray[onp.generic], array[:, dataon, tmin:tmax],
                    )
                )
            else:
                # else: array.shape = (num_nodes, feat_dim)
                return (
                    onp.array(null)
                    if len(dataon) == 0 else
                    cast(onpt.NDArray[onp.generic], array[:, dataon])
                )

        # Fetch inputs.
        # Node inputs will always be a 3D tensor with time axis, thus there is
        # no need to store exact snapshot range.
        node_feats_input_numpy = (
            fetch(
                self.node_feats, self.node_feat_input, self.dyn_node_feats,
                input_begin, input_end, 0.0,
            )
        )
        node_labels_input_numpy = (
            fetch(
                self.node_labels, self.node_label_input, self.dyn_node_labels,
                input_begin, input_end, 0,
            )
        )

        # Fetch targets.
        # Node targets will always be a 3D tensor with time axis, thus there is
        # no need to store exact snapshot range.
        # We define the range on time axis where each snapshot has exactly
        # length 1.

        # DynCSL self.node_labels.shape = (19, 1, 1600)
        # DynCSL self.node_feats.shape = (19, 2, 1600)
        # DynCSL self.node_times_abs.shape = (19, 1600)
        # DynCSL self.node_times_rel.shape = (19, 1600)
        # DynCSL self.node_times_inc.shape = (19, 1600)
        # DynCSL self.node_feat_input = [0, 1]
        # DynCSL self.node_feat_target = []
        # DynCSL self.node_label_input = []
        # DynCSL self.node_label_target = [0]
        # DynCSL self.dyn_node_feats = True
        # DynCSL self.dyn_node_labels = True
        # DynCSL self.timestamped_node_times = []
        # DynCSL self.num_nodes = 19
        # DynCSL self.num_times = 1600
        # DynCSL self.winaggr = 0
        # DynCSL self.window_history_size = 8
        # DynCSL self.window_future_size = 0
        # DynCSL self.WIN_SHIFT = 8

        node_feats_target_numpy  = (
            fetch(
                self.node_feats, self.node_feat_target, self.dyn_node_feats,
                target_begin, target_end, 0.0,
            )
        )
        node_labels_target_numpy = (
            fetch(
                self.node_labels, self.node_label_target, self.dyn_node_labels,
                target_begin, target_end, 0,
            )
        )
        # DynCSL node_feats_target_numpy = 0 
        # DynCSL node_labels_target_numpy.shape = (19, 1, 1)

        # Collect node timestamp data to be pinned.
        node_times_input_numpy_buf = []
        if self.ABS in self.timestamped_node_times:
            #
            node_times_input_numpy_buf.append(node_times_abs_input)
        if self.REL in self.timestamped_node_times:
            #
            node_times_input_numpy_buf.append(node_times_rel_input)
        if self.INC in self.timestamped_node_times:
            #
            node_times_input_numpy_buf.append(node_times_inc_input)
        if len(node_times_input_numpy_buf) == 0:
            # Use number of timestamps as place holder.
            node_times_input_numpy = onp.array(self.num_times)
        else:
            # Merge all timestamp data into a single array.
            # Put node axis at first for batching.
            node_times_input_numpy = onp.stack(node_times_input_numpy_buf)
            node_times_input_numpy = (
                onp.transpose(node_times_input_numpy, (1, 0, 2))
            )
        if len(self) == 1 and self.winaggr != self.WINAGG_NONE:
            #
            return (
                [
                    *self.cached, node_feats_input_numpy,
                    node_labels_input_numpy, node_times_input_numpy,
                ],
                [node_feats_target_numpy, node_labels_target_numpy],
            )
        else:
            #
            return (
                [
                    edge_tuples_numpy, edge_feats_numpy, edge_labels_numpy,
                    edge_ranges_numpy, edge_times_numpy, node_feats_input_numpy,
                    node_labels_input_numpy, node_times_input_numpy,
                ],
                [node_feats_target_numpy, node_labels_target_numpy],
            )

    def to_node_vecseq(self, /) -> VectorSequence:
        R"""
        Transform into node vector sequence metaset.
        """
        #
        raise RuntimeError

    def to_edge_vecseq(self, /) -> VectorSequence:
        R"""
        Transform into edge vector sequence metaset.
        """
        # UNEXPECT:
        # Temporal graph with static edge does not have edge vector sequence
        # data.
        raise NotImplementedError(
            "Temporal graph with static edge does not have edge vector "
            "sequence data.",
        )