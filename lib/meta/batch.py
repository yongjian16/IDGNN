R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import more_itertools as xitertools
from typing import List, Iterable, Optional, cast
from .meta import Meta
from .types import MEMOBJ, MEMBAT


def stack(samples: Iterable[List[MEMOBJ]], /) -> List[MEMOBJ]:
    R"""
    Stack sample arraies of the same column together.
    """
    #
    return (
        [
            onp.array(0, dtype=array.dtype) if array.ndim == 1 else array
            for array in (
                onp.stack(list(column))
                for column in xitertools.unzip(samples)
            )
        ]
    )


def batchize(
    metaset: Meta, meta_indices: List[int], meta_index_pad: Optional[int],
    meta_batch_size: int,
    /,
) -> MEMBAT:
    R"""
    Batch list of input and target array samples.
    """
    # A batch must have two paritions for input and target.
    (samples_input, samples_target) = (
        xitertools.unzip(
            metaset[sample]
            for sample in (
                meta_indices
                if meta_index_pad is None else
                xitertools
                .padded(meta_indices, meta_index_pad, meta_batch_size)
            )
        )
    )
    memory_input = stack(samples_input)
    memory_target = stack(samples_target)

    # Format memory
    flatten = (
        lambda memory: (
            list(
                map(
                    lambda array: (
                        cast(
                            MEMOBJ,
                            onp.reshape(
                                array,
                                (
                                    int(onp.prod(array.shape[:2])),
                                    *array.shape[2:],
                                ),
                            ),
                        )
                        if array.ndim > 1 else
                        cast(MEMOBJ, array)
                    ),
                    memory,
                )
            )
        )
    )
    memory_input = flatten(memory_input)
    memory_target = flatten(memory_target)
    return (memory_input, memory_target)


def batchize2(
    metaset: Meta, meta_indices: List[int], meta_index_pad: Optional[int],
    meta_batch_size: int,
    /,
) -> MEMBAT:
    R"""
    Batch list of input and target array samples.
    """
    #
    edge_tuples_buf = []
    edge_feats_buf = []
    edge_labels_buf = []
    edge_ranges_buf = []
    edge_times_buf = []
    node_feats_input_buf = []
    node_labels_input_buf = []
    node_times_input_buf = []
    node_feats_target_buf = []
    node_labels_target_buf = []
    num_nodes = 0
    num_edges = 0

    for sample in (
        meta_indices
        if meta_index_pad is None else
        xitertools.padded(meta_indices, meta_index_pad, meta_batch_size)
    ):
        #
        (
            [
                edge_tuples_sample, edge_feats_sample, edge_labels_sample,
                edge_ranges_sample, edge_times_sample, node_feats_input_sample,
                node_labels_input_sample, node_times_input_sample,
            ],
            [node_feats_target_sample, node_labels_target_sample],
        ) = metaset[sample]

        # Shift connectivity.
        edge_tuples_sample = onp.add(edge_tuples_sample, num_nodes)

        # Cache edge samples.
        edge_tuples_buf.append(edge_tuples_sample)
        edge_feats_buf.append(edge_feats_sample)
        edge_labels_buf.append(edge_labels_sample)
        edge_ranges_buf.append(edge_ranges_sample)
        edge_times_buf.append(edge_times_sample)

        # Cache node samples.
        node_feats_input_buf.append(node_feats_input_sample)
        node_labels_input_buf.append(node_labels_input_sample)
        node_times_input_buf.append(node_times_input_sample)
        node_feats_target_buf.append(node_feats_target_sample)
        node_labels_target_buf.append(node_labels_target_sample)

        # Accumulate.
        num_nodes = (
            num_nodes
            + (
                len(node_feats_input_sample)
                if node_feats_input_sample.ndim > 0 else
                len(node_labels_input_sample)
            )
        )
        num_edges = num_edges + edge_tuples_sample.shape[-1]
        # \\ print()
        # \\ print(num_nodes)
        # \\ print(num_edges)
        # \\ print(edge_tuples_sample.shape)
        # \\ print(edge_feats_sample.shape)
        # \\ print(edge_labels_sample.shape)
        # \\ print(edge_ranges_sample.shape)
        # \\ print(edge_times_sample.shape)
        # \\ print(node_feats_input_sample.shape)
        # \\ print(node_labels_input_sample.shape)
        # \\ print(node_times_input_sample.shape)
        # \\ print(node_feats_target_sample.shape)
        # \\ print(node_labels_target_sample.shape)

    # Stack edge data.
    # Edge timestamps are treated as feature sequences where timestamp kind
    # axis is the second axis (numeric axis 1).
    # This is different from static batching where timestamp kind is the first
    # axis.
    if edge_ranges_buf[0].ndim == 1:
        #
        edge_tuples_numpy = onp.concatenate(edge_tuples_buf, axis=1)
        if edge_feats_buf[0].ndim > 0:
            #
            edge_feats_numpy = onp.concatenate(edge_feats_buf)
        else:
            #
            edge_feats_numpy = onp.array(0.0)
        if edge_labels_buf[0].ndim > 0:
            #
            edge_labels_numpy = onp.concatenate(edge_labels_buf)
        else:
            #
            edge_labels_numpy = onp.array(0)
        edge_ranges_numpy = onp.array([[0], [num_edges]])
        edge_times_numpy = onp.concatenate(edge_times_buf, axis=0)
    elif edge_ranges_buf[0].ndim == 2:
        # We need to batch sample graphs into a large graph for each step.
        edge_ranges_numpy = onp.stack(edge_ranges_buf)
        edge_ranges_numpy = onp.sum(edge_ranges_numpy, axis=0)
        (_, num_times) = edge_ranges_sample.shape

        def dynedgebat(
            edge_indata_buf: List[onpt.NDArray[onp.generic]],
            /,
            *,
            axis: int,
        ) -> onpt.NDArray[onp.generic]:
            R"""
            Stack dynamic edge data.
            """
            #
            edge_outdata_buf: List[List[onpt.NDArray[onp.generic]]]

            #
            edge_outdata_buf = [[] for _ in range(num_times)]
            for (edge_ranges, edge_indata) in (
                zip(edge_ranges_buf, edge_indata_buf)
            ):
                #
                for (i, (edge_begin, edge_end)) in (
                    enumerate(zip(edge_ranges[0], edge_ranges[1]))
                ):
                    #
                    edge_outdata_buf[i].append(
                        onp.take(
                            edge_indata, range(edge_begin, edge_end),
                            axis=axis,
                        ),
                    )
            return (
                cast(
                    onpt.NDArray[onp.generic],
                    onp.concatenate(
                        [
                            onp.concatenate(edge_outdata, axis=axis)
                            for edge_outdata in edge_outdata_buf
                        ],
                        axis=axis,
                    ),
                )
            )

        #
        edge_tuples_numpy = dynedgebat(edge_tuples_buf, axis=1)
        if edge_feats_buf[0].ndim > 0:
            #
            edge_feats_numpy = dynedgebat(edge_feats_buf, axis=0)
        else:
            #
            edge_feats_numpy = onp.array(0.0)
        if edge_labels_buf[0].ndim > 0:
            #
            edge_labels_numpy = dynedgebat(edge_labels_buf, axis=0)
        else:
            #
            edge_labels_numpy = onp.array(0)
        if edge_times_buf[0].ndim > 0:
            #
            edge_times_numpy = dynedgebat(edge_times_buf, axis=1)
        else:
            #
            edge_times_numpy = onp.array(0.0)
    else:
        # UNEXPECT:
        # Unknown dynamic edge timestamp data type.
        raise NotImplementedError("Unknown dynamic edge timestamp data type.")

    # Stack node data.
    if node_feats_input_buf[0].ndim > 0:
        #
        node_feats_input_numpy = onp.concatenate(node_feats_input_buf)
    else:
        #
        node_feats_input_numpy = onp.array(0.0)
    if node_labels_input_buf[0].ndim > 0:
        #
        node_labels_input_numpy = onp.concatenate(node_labels_input_buf)
    else:
        #
        node_labels_input_numpy = onp.array(0)
    if node_times_input_buf[0].ndim > 0:
        #
        node_times_input_numpy = onp.concatenate(node_times_input_buf)
    else:
        #
        node_times_input_numpy = onp.array(0.0)
    if node_feats_target_buf[0].ndim > 0:
        #
        node_feats_target_numpy = onp.concatenate(node_feats_target_buf)
    else:
        #
        node_feats_target_numpy = onp.array(0.0)
    if node_labels_target_buf[0].ndim > 0:
        #
        node_labels_target_numpy = onp.concatenate(node_labels_target_buf)
    else:
        #
        node_labels_target_numpy = onp.array(0)
    # \\ print()
    # \\ print(edge_tuples_numpy.shape)
    # \\ print(edge_feats_numpy.shape)
    # \\ print(edge_labels_numpy.shape)
    # \\ print(edge_ranges_numpy.shape)
    # \\ print(edge_times_numpy.shape)
    # \\ print(node_feats_input_numpy.shape)
    # \\ print(node_labels_input_numpy.shape)
    # \\ print(node_times_input_numpy.shape)
    # \\ print(node_feats_target_numpy.shape)
    # \\ print(node_labels_target_numpy.shape)
    return (
        [
            edge_tuples_numpy, edge_feats_numpy, edge_labels_numpy,
            edge_ranges_numpy, edge_times_numpy, node_feats_input_numpy,
            node_labels_input_numpy, node_times_input_numpy,
        ],
        [node_feats_target_numpy, node_labels_target_numpy],
    )