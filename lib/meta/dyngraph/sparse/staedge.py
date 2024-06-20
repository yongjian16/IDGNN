R"""
"""
#
from os import times
import numpy as onp
import numpy.typing as onpt
from typing import List, Tuple, Union, cast
from ..dyngraph import DynamicGraph
from ...graph.sparse import AdjacencyList
from ...types import MEMPIN, MEMBAT
from ....utils.info import INFO, info5, infounion
from ...utils.normalize import normalize
from ...utils.repr.distrep import distrep
from ...utils.spindle import fitsplit
from ...vecseq import VectorSequence


class DynamicAdjacencyListStaticEdge(DynamicGraph, AdjacencyList):
    R"""
    Temporal graph as dynamic adjacency list meta whose edge connectivities,
    features and labels are static.
    """
    def shapize(self, /) -> None:
        R"""
        Get data shapes.
        """
        # Edge data is static as superclass.
        (self.num_edges, self.edge_feat_size) = self.edge_feats.shape
        (_, self.edge_label_size) = self.edge_labels.shape

        # Node data may have time axis which will requires explicitly setting
        # sliding window.
        (self.num_nodes, *self.node_feat_shape) = self.node_feats.shape
        (_, *self.node_label_shape) = self.node_labels.shape

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

        #
        self.edge_times_abs = (
            onp.tile(
                onp.reshape(self.timestamps, (1, self.num_times)),
                (self.num_edges, 1),
            )
        )
        self.edge_times_inc = (
            onp.concatenate(
                (
                    onp.zeros_like(
                        self.edge_times_abs,
                        shape=(self.num_edges, 1),
                    ),
                    self.edge_times_abs[:, 1:] - self.edge_times_abs[:, :-1],
                ),
                axis=1,
            )
        )
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

    def repr(self, /) -> INFO:
        R"""
        Get essential statistics for representation.
        """
        #
        repdict1 = DynamicGraph.repr(self)
        repdict2 = AdjacencyList.repr(self)
        return infounion(repdict1, repdict2)

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
        (indices_train, indices_valid, indices_test) = (
            fitsplit(
                {
                    "node": self.argsort_node_degree_in(),
                    "time": onp.arange(self.num_windows),
                }[spindle],
                list(proportion), list(priority),
            )
        )
        return (indices_train, indices_valid, indices_test)

    def reducesplit(
        self,
        proportion: Tuple[int, int, int], priority: Tuple[int, int, int],
        spindle: str, prop_keep: int, prop_total: int, inverse: bool,
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
        # \\ proportion = (
        # \\     [
        # \\         int(onp.floor(proportion[0] * prop_keep / prop_total)),
        # \\         proportion[0]
        # \\         - int(onp.floor(proportion[0] * prop_keep / prop_total)),
        # \\         proportion[1], proportion[2],
        # \\     ]
        # \\ )
        # \\ priority = (
        # \\     [
        # \\         priority[0] * 2, priority[0] * 2 + 1, priority[1] * 2,
        # \\         priority[2] * 2,
        # \\     ]
        # \\ )
        (indices_train, indices_valid, indices_test) = (
            fitsplit(
                {
                    "node": self.argsort_node_degree_in(),
                    "time": onp.arange(self.num_windows),
                }[spindle],
                list(proportion), list(priority),
            )
        )
        if spindle == "node":
            #
            map_indices_train = onp.argsort(-self.degree_ins[indices_train])
        elif spindle == "time":
            #
            map_indices_train = onp.arange(len(indices_train))
        else:
            # UNEXPECT:
            # Unknown split spindle.
            raise NotImplementedError(
                "Unknwon split spindle \"{:s}\".".format(spindle),
            )
        (map_indices_train_keep, map_indices_train_drop) = (
            fitsplit(
                map_indices_train, [prop_keep, prop_total - prop_keep], [1, 0],
            )
        )
        indices_train_keep = indices_train[map_indices_train_keep]
        indices_train_drop = indices_train[map_indices_train_drop]
        indices_train = (
            indices_train_drop if inverse else indices_train_keep
        )
        return (indices_train, indices_valid, indices_test)

    def normalizeby(
        self,
        indices: onpt.NDArray[onp.generic], spindle: str,
        /,
    ) -> List[List[Tuple[float, float]]]:
        R"""
        Normalize data using statistics from indices.
        """
        # We can normalize all edge features directly since they are static
        # over time.
        (self.edge_feats, edge_factors) = (
            normalize(self.edge_feats, self.edge_feats, axis=0)
        )

        # We may need to additionally normalize on time dimense for node
        # features.
        vdim = {"node": 0, "time": 2}[spindle]
        axis = (0, 2) if self.dyn_node_feats else (0,)
        (self.node_feats, node_factors) = (
            normalize(
                self.node_feats,
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
                                distrep(self.edge_feats, axis=0, n=n)[0],
                                self.meaningless_edge_feats,
                            )
                        ),
                        "Edge Label": (
                            fade(
                                distrep(self.edge_labels, axis=0, n=n)[0],
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
        #
        (
            edge_tuples_numpy, edge_feats_numpy, edge_labels_numpy,
        ) = AdjacencyList.pin(self, batch_size)

        # Dynamic graph with static edge has only one edge snapshot rather than
        # multiple steps of snapshots.
        (_, num_edges) = edge_tuples_numpy.shape
        edge_ranges_numpy = onp.array([[0], [num_edges]])
        edge_ranges_numpy = (
            onp.tile(edge_ranges_numpy, (1, self.window_history_size))
        )

        # For static edge, relative and incremental timestamps will always be
        # the same for different windows.
        edge_times_abs = self.edge_times_abs[:, :self.window_history_size]
        edge_times_rel = (
            onp.reshape(edge_times_abs[:, -1], (self.num_edges, 1))
            - edge_times_abs
        )
        edge_times_inc = self.edge_times_inc[:, :self.window_history_size]

        # We can not pin absolute timestamps in memory since it is dynamic 
        # w.r.t. each batch.
        if self.ABS in self.timestamped_edge_times:
            # UNEXPECT:
            # Absolute timestamps should not be pinned.
            raise NotImplementedError(
                "Absolute edge timestamps should not be pinned."
            )

        # Collect edge timestamp data to be pinned.
        edge_times_numpy_buf = []
        if self.ABS in self.timestamped_edge_times:
            #
            edge_times_numpy_buf.append(edge_times_abs)
        if self.REL in self.timestamped_edge_times:
            #
            edge_times_numpy_buf.append(edge_times_rel)
        if self.INC in self.timestamped_edge_times:
            #
            edge_times_numpy_buf.append(edge_times_inc)
        if len(edge_times_numpy_buf) == 0:
            # Use number of timestamps as place holder.
            edge_times_numpy = onp.array(self.num_times)
        else:
            # Merge all timestamp data into a single array.
            # Extend by batch size.
            edge_times_numpy = onp.stack(edge_times_numpy_buf)
            edge_times_numpy = onp.tile(edge_times_numpy, (1, batch_size, 1))

            # Ensure timestamp data can directly match with other edge data.
            num_edges = batch_size * self.num_edges
            edge_tuples_numpy = (
                onp.tile(
                    onp.reshape(edge_tuples_numpy, (2, 1, num_edges)),
                    (1, self.window_history_size, 1),
                )
            )
            edge_tuples_numpy = (
                onp.reshape(
                    edge_tuples_numpy,
                    (2, self.window_history_size * num_edges),
                )
            )
            if edge_feats_numpy.ndim > 0:
                #
                edge_feats_numpy = (
                    onp.tile(
                        onp.reshape(
                            edge_feats_numpy,
                            (1, num_edges, self.edge_feat_size),
                        ),
                        (self.window_history_size, 1, 1),
                    )
                )
                edge_feats_numpy = (
                    onp.reshape(
                        edge_feats_numpy,
                        (
                            self.window_history_size * num_edges,
                            self.edge_feat_size,
                        ),
                    )
                )
            if edge_labels_numpy.ndim > 0:
                #
                edge_labels_numpy = (
                    onp.tile(
                        onp.reshape(
                            edge_labels_numpy,
                            (1, num_edges, self.edge_label_size),
                        ),
                        (self.window_history_size, 1, 1),
                    )
                )
                edge_labels_numpy = (
                    onp.reshape(
                        edge_labels_numpy,
                        (
                            self.window_history_size * num_edges,
                            self.edge_label_size,
                        ),
                    )
                )
            edge_ranges_numpy = (
                edge_ranges_numpy
                + (
                    onp.reshape(
                        onp.arange(self.window_history_size) * num_edges,
                        (1, self.window_history_size),
                    )
                )
            )
            edge_times_numpy = (
                onp.reshape(
                    onp.transpose(edge_times_numpy, (0, 2, 1)),
                    (
                        len(edge_times_numpy),
                        self.window_history_size * num_edges,
                    ),
                )
            )
        return (
            [
                edge_tuples_numpy, edge_feats_numpy, edge_labels_numpy,
                edge_ranges_numpy, edge_times_numpy,
            ]
        )

    def __len__(self, /) -> int:
        R"""
        Length of the class.
        """
        #
        return self.num_windows

    def idx_to_timesteps(self, idx: int):
        input_tie1 = idx
        input_tie2 = input_tie1 + self.window_history_size
        target_tie1 = input_tie2
        target_tie2 = target_tie1 + self.window_future_size
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
        input_tie1 = idx
        input_tie2 = input_tie1 + self.window_history_size
        target_tie1 = input_tie2
        target_tie2 = target_tie1 + self.window_future_size
        input_begin = min(input_tie1, input_tie2)
        input_end = max(input_tie1, input_tie2)
        target_begin = min(target_tie1, target_tie2)
        target_end = max(target_tie1, target_tie2)

        # Get timestamp data of current batch.
        node_times_abs_input = self.node_times_abs[:, input_begin:input_end]
        node_times_rel_input = (
            onp.reshape(node_times_abs_input[:, -1], (self.num_nodes, 1))
            - node_times_abs_input
        )
        node_times_inc_input = self.node_times_inc[:, input_begin:input_end]

        def fetch(
            array: onpt.NDArray[onp.generic], dataon: List[int], dynamic: bool,
            tmin: int, tmax: int, null: Union[int, float],
        ) -> onpt.NDArray[onp.generic]:
            R"""
            Fetch online memory from data.
            """
            if dynamic:
                #
                return (
                    onp.array(null)
                    if len(dataon) == 0 else
                    cast(
                        onpt.NDArray[onp.generic], array[:, dataon, tmin:tmax],
                    )
                )
            else:
                #
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
        return (
            [
                node_feats_input_numpy, node_labels_input_numpy,
                node_times_input_numpy,
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