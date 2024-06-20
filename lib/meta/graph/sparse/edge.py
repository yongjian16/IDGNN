R"""
"""
#
import numpy as onp
import numpy.typing as onpt
from .types import EDGEONLY


def edge_symmetrize(
    edge_srcs: onpt.NDArray[onp.generic], edge_dsts: onpt.NDArray[onp.generic],
    edge_feats: onpt.NDArray[onp.generic],
    edge_labels: onpt.NDArray[onp.generic],
    /,
) -> EDGEONLY:
    R"""
    Symmetrize edge data.
    """
    # Collect edges in one direction.
    edge_srcs1 = edge_srcs
    edge_dsts1 = edge_dsts
    edge_feats1 = edge_feats
    edge_labels1 = edge_labels

    # Collect edges in the other direction.
    edge_srcs2 = edge_dsts
    edge_dsts2 = edge_srcs
    edge_feats2 = edge_feats
    edge_labels2 = edge_labels

    # Concatenate directly.
    return (
        onp.concatenate((edge_srcs1, edge_srcs2), axis=0),
        onp.concatenate((edge_dsts1, edge_dsts2), axis=0),
        onp.concatenate((edge_feats1, edge_feats2), axis=0),
        onp.concatenate((edge_labels1, edge_labels2), axis=0),
    )


def edge_sort(
    edge_srcs: onpt.NDArray[onp.generic], edge_dsts: onpt.NDArray[onp.generic],
    edge_feats: onpt.NDArray[onp.generic],
    edge_labels: onpt.NDArray[onp.generic],
    /,
) -> EDGEONLY:
    R"""
    Sort edge data by source-then-destination node IDs.
    """
    # Enforce edge indices range for sorting safety.
    edge_key1 = edge_srcs
    edge_key2 = edge_dsts
    node_max = int(onp.floor(onp.sqrt(onp.iinfo(onp.int64).max)))
    if (
        onp.any(onp.greater_equal(edge_key1, node_max))
        or onp.any(onp.greater_equal(edge_key2, node_max))
    ):
        # It is possible to have improper arguments.
        raise RuntimeError(
            "Edge indices are too large (square exceeds int64) to be sorted.",
        )

    # Sort.
    edge_key = onp.multiply(edge_key1, node_max) + edge_key2
    edge_indices = onp.argsort(edge_key)
    return (
        edge_srcs[edge_indices], edge_dsts[edge_indices],
        edge_feats[edge_indices], edge_labels[edge_indices],
    )


def edge_unique(
    edge_srcs: onpt.NDArray[onp.generic], edge_dsts: onpt.NDArray[onp.generic],
    /,
    *,
    sorted: bool,
) -> bool:
    R"""
    Ensure edge uniqueness on sort-then-destination node IDs.
    """
    # Safety check.
    if not sorted:
        # It is possible to have improper arguments.
        raise RuntimeError("Edge must be sorted before checking duplication.")

    #
    dup_srcs = edge_srcs[:-1] == edge_srcs[1:]
    dup_dsts = edge_dsts[:-1] == edge_dsts[1:]
    dup_edges = dup_srcs & dup_dsts
    return onp.any(dup_edges).item()