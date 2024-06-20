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
from ...meta.vecseq import VectorSequence
from ..types import TIMECOST
from ..transfer import transfer


class FrameworkVectorSequence(FrameworkIndexable[VectorSequence]):
    R"""
    Framework with vector sequence meta samples.
    """
    #
    BATCH_PAD = False

    def seq_batch(
        self,
        meta_indices: List[int], meta_index_pad: Optional[int],
        meta_batch_size: int,
        /,
    ) -> List[onpt.NDArray[onp.generic]]:
        R"""
        Construct a batch by sequence data.
        """
        #
        buf_feat = []
        for idx in meta_indices:
            #
            ((sample_feat,), _) = self.metaset[idx]
            buf_feat.append(sample_feat)
        feats = onp.stack(buf_feat, axis=1)
        return [feats]

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

        # Get sequence batch.
        timeparts["generate"] = []
        timeparts["transfer"] = []
        timeparts["forward"] = []
        timeparts["backward"] = []
        batch_indices = meta_indices
        for batch in xitertools.chunked(batch_indices, meta_batch_size):
            # Batchize only nodes of batch graphs.
            elapsed = time.time()
            memory_numpy = (
                self.seq_batch(list(batch), meta_index_pad, meta_batch_size)
            )
            cast(List[float], timeparts["generate"]).append(
                time.time() - elapsed,
            )

            # Node mask need special processing.
            elapsed = time.time()
            memory_ondev = transfer(memory_numpy, self.device)
            cast(List[float], timeparts["transfer"]).append(
                time.time() - elapsed,
            )

            # Rearange and reshape device memory tensors to fit task
            # requirements.
            (memory_input_ondev, memory_target_ondev) = (
                self.neuralnet.reshape(pinned, memory_ondev)
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
                self.neuralnet.sidestep(
                    *memory_output_ondev, *memory_target_ondev,
                )
                self.gradclip(self.neuralnet, 1.0)
                self.optim.step()
            cast(List[float], timeparts["backward"]).append(
                time.time() - elapsed,
            )
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

        # Get sequence batch.
        timeparts["generate"] = []
        timeparts["transfer"] = []
        timeparts["forward"] = []
        batch_indices = meta_indices
        for batch in xitertools.chunked(batch_indices, meta_batch_size):
            # Batchize only nodes of batch graphs.
            elapsed = time.time()
            memory_numpy = (
                self.seq_batch(list(batch), meta_index_pad, meta_batch_size)
            )
            cast(List[float], timeparts["generate"]).append(
                time.time() - elapsed,
            )

            # Node mask need special processing.
            elapsed = time.time()
            memory_ondev = transfer(memory_numpy, self.device)
            cast(List[float], timeparts["transfer"]).append(
                time.time() - elapsed,
            )

            # Rearange and reshape device memory tensors to fit task
            # requirements.
            (memory_input_ondev, memory_target_ondev) = (
                self.neuralnet.reshape(pinned, memory_ondev)
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