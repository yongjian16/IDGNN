R"""
"""
#
import abc
import numpy as onp
import numpy.typing as onpt
from typing import Callable, Union, List
from ..meta import Meta
from ..utils.dataon import dataon
from ...utils.info import INFO
from ..utils.repr.shape import shape5


class Graph(Meta):
    R"""
    Graph meta.
    """
    def __annotation__(self, /) -> None:
        R"""
        Annotate for class instance attributes.
        """
        #
        self.num_nodes: int
        self.num_edges: int

        #
        self.edge_feat_size: int
        self.edge_label_size: int
        self.node_feat_size: int
        self.node_label_size: int

        #
        self.meaningless_edge_feats: bool
        self.meaningless_edge_labels: bool
        self.meaningless_node_feats: bool
        self.meaningless_node_labels: bool

    @abc.abstractmethod
    def shapize(self, /) -> None:
        R"""
        Get data shapes.
        """
        #
        ...

    @abc.abstractmethod
    def collect(self, /) -> None:
        R"""
        Collect essential statistics.
        """
        #
        ...

    def repr(self, /) -> INFO:
        R"""
        Get essential statistics for representation.
        """
        #
        repdict = Meta.repr(self)
        repdict["Basic"]["#Nodes"] = str(self.num_nodes)
        repdict["Basic"]["#Edges"] = str(self.num_edges)
        repdict["(Feat)ure"] = (
            {
                "Edge Feat Size": (
                    shape5(
                        str(self.edge_feat_size), self.edge_feat_size,
                        "float64", self.meaningless_edge_feats,
                    )
                ),
                "Edge Label Size": (
                    shape5(
                        str(self.edge_label_size), self.edge_label_size,
                        "int64", self.meaningless_edge_labels,
                    )
                ),
                "Node Feat Size": (
                    shape5(
                        str(self.node_feat_size), self.node_feat_size,
                        "float64", self.meaningless_node_feats,
                    )
                ),
                "Node Label Size": (
                    shape5(
                        str(self.node_label_size), self.node_label_size,
                        "int64", self.meaningless_node_labels,
                    )
                ),
            }
        )
        repdict["(Deg)ree"] = (
            {
                "Deg In": "",
                "Deg Out": "",
                "Deg In Dist": "",
                "Deg Out Dist": "",
            }
        )
        return repdict

    @abc.abstractmethod
    def argsort_node_degree_in(self, /) -> onpt.NDArray[onp.generic]:
        R"""
        Argsort nodes by in degrees (large to small).
        """
        #
        ...

    @abc.abstractmethod
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
        ...

    def inputon(self, on: List[Union[str, List[int]]]) -> None:
        R"""
        Set input feature and label data columns.
        """
        #
        (on_edge_feat, on_edge_label, on_node_feat, on_node_label) = on
        self.edge_feat_input = (
            dataon(
                on_edge_feat, self.edge_feat_size, self.meaningless_edge_feats,
            )
        )
        self.edge_label_input = (
            dataon(
                on_edge_label, self.edge_label_size,
                self.meaningless_edge_labels,
            )
        )
        self.node_feat_input = (
            dataon(
                on_node_feat, self.node_feat_size, self.meaningless_node_feats,
            )
        )
        self.node_label_input = (
            dataon(
                on_node_label, self.node_label_size,
                self.meaningless_node_labels,
            )
        )

    def targeton(self, on: List[Union[str, List[int]]]) -> None:
        R"""
        Set tatget feature and label data columns.
        """
        #
        (on_edge_feat, on_edge_label, on_node_feat, on_node_label) = on
        self.edge_feat_target = (
            dataon(
                on_edge_feat, self.edge_feat_size, self.meaningless_edge_feats,
            )
        )
        self.edge_label_target = (
            dataon(
                on_edge_label, self.edge_label_size,
                self.meaningless_edge_labels,
            )
        )
        self.node_feat_target = (
            dataon(
                on_node_feat, self.node_feat_size, self.meaningless_node_feats,
            )
        )
        self.node_label_target = (
            dataon(
                on_node_label, self.node_label_size,
                self.meaningless_node_labels,
            )
        )