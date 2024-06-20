R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import time
import torch
import more_itertools as xitertools
from typing import List, Tuple, Optional, Union, cast
from ..indexable import FrameworkIndexable
from ...meta.dyngraph.sparse.staedge import DynamicAdjacencyListStaticEdge
from ...meta.dyngraph.sparse.dynedge import DynamicAdjacencyListDynamicEdge
from ..types import TIMECOST
from ..transfer import transfer
from ...meta.batch import batchize, batchize2


class FrameworkDynamicGraph(
    FrameworkIndexable[
        Union[DynamicAdjacencyListStaticEdge, DynamicAdjacencyListDynamicEdge],
    ],
):
    R"""
    Framework with dynamic graph meta samples.
    """
    #
    BATCH_PAD = True

    def nodesplit_masks(
        self,
        meta_indices: List[int], meta_batch_size: int,
        /,
    ) -> onpt.NDArray[onp.generic]:
        R"""
        Translate given metaset indices into available node indices.
        For node pindle, only given metaset indices will be used.
        For time pindle, all node indices will be used.
        """
        #
        if self.metaspindle == "node":
            # If spindle is node, only nodes of given indices will be
            # available.
            masks_numpy = (
                onp.zeros((self.metaset.num_nodes,)).astype(onp.int64)
            )
            masks_numpy[meta_indices] = 1
            masks_numpy = onp.tile(masks_numpy, (meta_batch_size,))
        else:
            # Otherwise, all nodes are avaiable.
            masks_numpy = (
                onp.ones((meta_batch_size * self.metaset.num_nodes,))
                .astype(onp.int64)
            )
        return masks_numpy

    def set_node_batching(self, with_edge: bool, /) -> None:
        R"""
        Set batch construction.
        """
        #
        self.node_batch_with_edge = with_edge

    def node_batch(
        self,
        meta_indices: List[int], meta_index_pad: Optional[int],
        meta_batch_size: int,
        /,
    ) -> List[onpt.NDArray[onp.generic]]:
        R"""
        Construct a batch by node data.
        """
        # Fill the batch to constant batch size by given padding index.
        # Create a mask over samples to filter padded ones in later usage.
        # Expand and apply the same mask as samples to all nodes in
        # corresponding samples.
        masks_numpy = onp.zeros((meta_batch_size,)).astype(onp.int64)
        masks_numpy[:len(meta_indices)] = 1
        masks_numpy = onp.repeat(masks_numpy, self.metaset.num_nodes, axis=0)
        
        # Get memory.
        if self.node_batch_with_edge:
            #
            (memory_input_numpy, memory_target_numpy) = batchize2(
                self.metaset, meta_indices, meta_index_pad, meta_batch_size,
            )
        else:
            #
            (memory_input_numpy, memory_target_numpy) = batchize(
                self.metaset, meta_indices, meta_index_pad, meta_batch_size,
            )
        return [masks_numpy, *memory_input_numpy, *memory_target_numpy]
    
    def train(
        self,
        meta_indices: List[int], meta_index_pad: Optional[int],
        meta_batch_size: int, pinned: List[torch.Tensor],
        /,
    ) -> TIMECOST:
        R"""
        Train.
        Mostly used for neural network parameter tuning.
        """
        #
        timeparts: TIMECOST

        #
        timeparts = {}

        #
        elapsed = time.time()
        masks_nodesplit_numpy = (
            self.nodesplit_masks(meta_indices, meta_batch_size)
        )
        timeparts["generate"] = [time.time() - elapsed]
        elapsed = time.time()
        (masks_nodesplit_ondev,) = (
            transfer([masks_nodesplit_numpy], self.device)
        )
        timeparts["transfer"] = [time.time() - elapsed]

        # If we split data by node, given indices is indeed transductive node
        # indices which has been converted into a mask array before.
        # Thus, we will batch over the full metaset.
        # Otherwise, we only batch over metaset of given meta indices.
        if self.metaspindle == "node":
            #
            batch_indices = list(range(len(self.metaset)))
        else:
            #
            batch_indices = meta_indices

        #
        timeparts["forward"] = []
        timeparts["backward"] = []
        avg_loss = 0.
        cnt = 0

        for batch in xitertools.chunked(batch_indices, meta_batch_size):
            # Batchize only nodes of batch graphs.
            elapsed = time.time()
            memory_node_numpy = (
                self.node_batch(list(batch), meta_index_pad, meta_batch_size)
            )
            cast(List[float], timeparts["generate"]).append(
                time.time() - elapsed,
            )

            # Node mask need special processing.
            elapsed = time.time()
            (masks_hole_ondev, *memory_node_ondev) = (
                transfer(memory_node_numpy, self.device)
            )
            cast(List[float], timeparts["transfer"]).append(
                time.time() - elapsed,
            )
            node_masks_ondev = masks_hole_ondev * masks_nodesplit_ondev

            # Rearange and reshape device memory tensors to fit task
            # requirements.
            (memory_input_ondev, memory_target_ondev) = (
                self.neuralnet.reshape(
                    pinned, memory_node_ondev, node_masks_ondev,
                )
            )

            # Forward.
            elapsed = time.time()
            memory_output_ondev = self.neuralnet.forward(*memory_input_ondev)
            cast(List[float], timeparts["forward"]).append(
                time.time() - elapsed,
            )

            # Backward.
            elapsed = time.time()
            if self.neuralnet.num_resetted_params > 0:
                #
                self.optim.zero_grad()
                loss = self.neuralnet.sidestep(
                    *memory_output_ondev, *memory_target_ondev,
                )
                self.gradclip(self.neuralnet, 1.0)
                self.optim.step()
                avg_loss += loss.item()
                cnt += 1

            cast(List[float], timeparts["backward"]).append(
                time.time() - elapsed,
            )
        print(f'Loss: {avg_loss/cnt:.4f}')
        return timeparts

    def evaluate(
        self,
        meta_indices: List[int], meta_index_pad: Optional[int],
        meta_batch_size: int, pinned: List[torch.Tensor],
        /,
    ) -> Tuple[List[float], TIMECOST]:
        R"""
        Evaluate.
        Mostly used for neural network parameter evaluation.
        """
        #
        timeparts: TIMECOST

        #
        timeparts = {}
        estimates = []

        #
        elapsed = time.time()
        masks_nodesplit_numpy = (
            self.nodesplit_masks(meta_indices, meta_batch_size)
        )
        timeparts["generate"] = []
        elapsed = time.time()
        (masks_nodesplit_ondev,) = (
            transfer([masks_nodesplit_numpy], self.device)
        )
        timeparts["transfer"] = [time.time() - elapsed]

        # If we split data by node, given indices is indeed transductive node
        # indices which has been converted into a mask array before.
        # Thus, we will batch over the full metaset.
        # Otherwise, we only batch over metaset of given meta indices.
        if self.metaspindle == "node":
            #
            batch_indices = list(range(len(self.metaset)))
        else:
            #
            batch_indices = meta_indices

        #
        timeparts["forward"] = []
        # \\ cnt = 0
        # \\ self.neuralnet.SEE_EMBEDS = True
        for batch in xitertools.chunked(batch_indices, meta_batch_size):
            # Batchize only nodes of batch graphs.
            elapsed = time.time()
            memory_node_numpy = (
                self.node_batch(list(batch), meta_index_pad, meta_batch_size)
            )
            cast(List[float], timeparts["generate"]).append(
                time.time() - elapsed,
            )

            # Node mask need special processing.
            elapsed = time.time()
            (masks_hole_ondev, *memory_node_ondev) = (
                transfer(memory_node_numpy, self.device)
            )
            cast(List[float], timeparts["transfer"]).append(
                time.time() - elapsed,
            )
            node_masks_ondev = masks_hole_ondev * masks_nodesplit_ondev

            # Rearange and reshape device memory tensors to fit task
            # requirements.
            (memory_input_ondev, memory_target_ondev) = (
                self.neuralnet.reshape(
                    pinned, memory_node_ondev, node_masks_ondev,
                )
            )

            # Forward.
            elapsed = time.time()
            memory_output_ondev = self.neuralnet.forward(*memory_input_ondev)
            cast(List[float], timeparts["forward"]).append(
                time.time() - elapsed,
            )
            # Performance metrics.
            estimates.append(
                self.neuralnet.metrics(
                    *memory_output_ondev, *memory_target_ondev,
                ),
            )
            # \\ cnt += 1
            # \\ if cnt == 2:
            # \\     #
            # \\     self.neuralnet.SEE_EMBEDS = False
        # \\ self.neuralnet.SEE_EMBEDS = False

        # Collect mean of all metrics and time costs.
        return (
            [
                sum(measure for (_, measure) in record)
                / sum(size for (size, _) in record)
                for record in (
                    [list(metric) for metric in xitertools.unzip(estimates)]
                )
            ],
            timeparts,
        )