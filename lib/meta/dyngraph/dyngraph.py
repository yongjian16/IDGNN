R"""
"""
#
import abc
import numpy as onp
import numpy.typing as onpt
from typing import Optional, List
from ..graph.graph import Graph
from ...utils.info import INFO
from ..utils.slidewin import sliding_window
from ..vecseq import VectorSequence


class DynamicGraph(Graph):
    R"""
    Dynamic graph meta.
    """
    # Timestamp kinds.
    ABS = 0
    REL = 1
    INC = 2
    TIMESTAMP_INT = {"abs": ABS, "rel": REL, "inc": INC}
    TIMESTAMP_MSG = {ABS: "absolute", REL: "relative", INC: "incremental"}

    def __annotation__(self, /) -> None:
        R"""
        Annotate for class instance attributes.
        """
        #
        self.num_times: int
        self.num_windows: int

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
        # Always use float timestamps and match with time steps one-by-one.
        # We treat timestamps differently from features since we do not want
        # to automatically scale timestamps.
        # It should be manually normalized before given so that timestamps are
        # meaningful.
        self.timestamps = (
            onp.reshape(timestamps, (self.num_times,)).astype(onp.float64)
        )

        # Save timestamping flags.
        self.timestamped_edge_times = (
            [
                self.TIMESTAMP_INT[timekind]
                for timekind in timestamped_edge_times
            ]
        )
        self.timestamped_node_times = (
            [
                self.TIMESTAMP_INT[timekind]
                for timekind in timestamped_node_times
            ]
        )
        self.timestamped_edge_feats = (
            [
                self.TIMESTAMP_INT[timekind]
                for timekind in timestamped_edge_feats
            ]
        )
        self.timestamped_node_feats = (
            [
                self.TIMESTAMP_INT[timekind]
                for timekind in timestamped_node_feats
            ]
        )

        # TODO:
        # Support timestamps being raw features.
        if (
            len(self.timestamped_edge_feats) > 0
            or len(self.timestamped_node_feats) > 0
        ):
            # UNEXPECT:
            # Timestamps as raw features are not supported.
            raise NotImplementedError(
                "Timestamps as raw edge or node features are not supported.",
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
        self.num_windows = self.num_times - self.window_size + 1

    def repr(self, /) -> INFO:
        R"""
        Get essential statistics for representation.
        """
        #
        repdict = Graph.repr(self)
        repdict["Basic"]["#Times(TIDs)"] = str(self.num_times)
        repdict["Basic"]["TID.<>Edge"] = (
            "[{:s}]".format(
                ", ".join(
                    self.TIMESTAMP_MSG[timeint]
                    for timeint in self.timestamped_edge_times
                )
            )
        )
        repdict["Basic"]["TID.||Edge"] = (
            "[{:s}]".format(
                ", ".join(
                    self.TIMESTAMP_MSG[timeint]
                    for timeint in self.timestamped_edge_feats
                )
            )
        )
        repdict["Basic"]["TID.<>Node"] = (
            "[{:s}]".format(
                ", ".join(
                    self.TIMESTAMP_MSG[timeint]
                    for timeint in self.timestamped_node_times
                )
            )
        )
        repdict["Basic"]["TID.||Node"] = (
            "[{:s}]".format(
                ", ".join(
                    self.TIMESTAMP_MSG[timeint]
                    for timeint in self.timestamped_node_feats
                )
            )
        )
        repdict["Basic"]["#Windows"] = str(self.num_windows)
        repdict["Basic"]["Win.Obsv"] = str(self.window_history_size)
        repdict["Basic"]["Win.Pred"] = str(self.window_future_size)
        return repdict

    @abc.abstractmethod
    def to_node_vecseq(self, /) -> VectorSequence:
        R"""
        Transform into node vector sequence metaset.
        """
        #
        ...

    @abc.abstractmethod
    def to_edge_vecseq(self, /) -> VectorSequence:
        R"""
        Transform into edge vector sequence metaset.
        """
        #
        ...