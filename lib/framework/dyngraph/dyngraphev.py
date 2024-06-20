R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import time
import torch
import more_itertools as xitertools
import os
from typing import List, Tuple, Optional, Union, cast
from .dyngraph import FrameworkDynamicGraph
from ...meta.dyngraph.sparse.staedge import DynamicAdjacencyListStaticEdge
from ...meta.dyngraph.sparse.dynedge import DynamicAdjacencyListDynamicEdge
from ..types import TIMECOST
from ..transfer import transfer
from ...meta.batch import batchize, batchize2
from ...task.dyncsl import GraphWindowClassification
from ...task.classification import metrics
from ..indexable import DIR


class FrameworkDynamicGraphForEval(FrameworkDynamicGraph):
    R"""
    Framework with dynamic graph meta samples.
    """
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
        # EXPECT:
        # Reject training by this framework.
        raise RuntimeError

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
        if not isinstance(self.neuralnet, GraphWindowClassification):
            # EXPECT:
            # Reject not-graph-classification by this framework.
            raise RuntimeError

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
        output_buf = []
        target_buf = []
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

            #
            (graph_output_feats,) = memory_output_ondev
            (_, node_target_labels, _) = memory_target_ondev
            graph_target_labels = (
                torch.reshape(
                    node_target_labels.squeeze(),
                    (len(graph_output_feats), 19),
                )[:, 0]
            )
            output_buf.append(graph_output_feats)
            target_buf.append(graph_target_labels)
        graph_output_feats = torch.cat(output_buf)
        graph_target_labels = torch.cat(target_buf)

        # Performance metrics.
        estimates.append(
            metrics(
                graph_output_feats, graph_target_labels,
                cast(torch.Tensor, self.neuralnet.label_weights)
            )
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

    def besteval(
        self,
        proportion: Tuple[int, int, int], priority: Tuple[int, int, int],
        train_prop: Tuple[int, int, bool],
        /,
        batch_size: int, validon: int, validrep: str, resume: str,
    ) -> None:
        R"""
        Best test after training.
        """
        #
        self.ptnnp = os.path.join(resume, "{:s}.ptnnp".format(self.identifier))
        self.ptlog = os.path.join(resume, "{:s}.ptlog".format(self.identifier))

        # Ensure the existence of log directory.
        if not os.path.isdir(DIR):
            #
            os.makedirs(DIR, exist_ok=True)
        self.ptbev = os.path.join(DIR, "{:s}.ptbev".format(self.identifier))

        #
        self.preprocess(proportion, priority, train_prop)

        #
        meta_index_pad = (
            onp.min(self.meta_indices_train).item() if self.BATCH_PAD else None
        )

        # Pin shared memory.
        pinned_numpy = self.metaset.pin(batch_size)
        pinned_ondev = transfer(pinned_numpy, self.device)

        # Best test after training.
        print("=" * 10 + " " + "Test (best)" + " " + "=" * 10)
        self.neuralnet.load_state_dict(torch.load(self.ptnnp))
        # \\ (_, _, valid_orig, test_orig, _, _) = torch.load(self.ptlog)
        with torch.no_grad():
            #
            (metrics_valid, _) = (
                self.evaluate(
                    self.meta_indices_valid.tolist(), meta_index_pad,
                    batch_size, pinned_ondev,
                )
            )
            (metrics_test, _) = (
                self.evaluate(
                    self.meta_indices_test.tolist(), meta_index_pad,
                    batch_size, pinned_ondev,
                )
            )
        # \\ valid_orig = (*valid_orig[:-1], metrics_valid[-1])
        # \\ test_orig = (*test_orig[:-1], metrics_test[-1])
        print(
            "Valid\x1b[94m:\x1b[0m \x1b[3m{:s}\x1b[0m: {:s}"
            .format(validrep, "{:.6f}".format(metrics_valid[validon])[:8]),
        )
        print(
            " Test\x1b[94m:\x1b[0m \x1b[3m{:s}\x1b[0m: {:s}"
            .format(validrep, "{:.6f}".format(metrics_test[validon])[:8]),
        )

        #
        print("=" * 10 + " " + "Relog" + " " + "=" * 10)
        (factors, logs, _, _, gpu_mem_peak, timecosts) = torch.load(self.ptlog)
        torch.save(
            (
                factors, logs, metrics_valid[validon], metrics_test,
                gpu_mem_peak, timecosts,
            ),
            self.ptbev,
        )