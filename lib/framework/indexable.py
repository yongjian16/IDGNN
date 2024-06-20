R"""
"""
#
import abc
import numpy as onp
import torch
import time
import os
from typing import List, Tuple, Optional, Generic, TypeVar
from ..meta.meta import Meta
from .transfer import transfer
from .types import TIMECOST
from ..task.task import Task
from ..utils.info import info5
from .gradclip import GRADCLIPS
from tqdm import tqdm


#
META = TypeVar("META", bound=Meta)


#
DIR = "log"
LR_DECAY = 10
LR_THRES = 1e-4 * (1 + 1 / LR_DECAY)
IMP_ABS = 0
IMP_REL = 0


class FrameworkIndexable(abc.ABC, Generic[META]):
    R"""
    Framework with indexable (by finite integer) meta samples.
    """
    #
    BATCH_PAD: bool

    def __init__(
        self,
        identifier: str, metaset: META, neuralnet: Task,
        /,
        *,
        lr: float, weight_decay: float, seed: int, device: str,
        metaspindle: str, gradclip: str,
    ) -> None:
        R"""
        Initalize the class.
        """
        #
        self.identifier = identifier
        self.metaset = metaset
        self.neuralnet = neuralnet
        self.seed = seed
        self.device = device

        #
        self.metaspindle = metaspindle

        #
        if neuralnet.num_resetted_params > 0:
            #
            self.gradclip = GRADCLIPS[gradclip]
            self.optim = (
                torch.optim.Adam(
                    self.neuralnet.parameters(),
                    lr=lr, weight_decay=weight_decay,
                )
            )

        # Move model to device after creating the optimizer.
        self.neuralnet = neuralnet.to(device)

        # Ensure the existence of log directory.
        if not os.path.isdir(DIR):
            #
            os.makedirs(DIR, exist_ok=True)
        self.ptnnp = os.path.join(DIR, "{:s}.ptnnp".format(self.identifier))
        self.ptres = os.path.join(DIR, "{:s}.ptres".format(self.identifier))
        self.ptlog = os.path.join(DIR, "{:s}.ptlog".format(self.identifier))

    @abc.abstractmethod
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
        ...

    @abc.abstractmethod
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
        ...

    def preprocess(
        self,
        proportion: Tuple[int, int, int], priority: Tuple[int, int, int],
        train_prop: Tuple[int, int, bool],
        /,
    ) -> None:
        R"""
        Preprocess metaset before fitting.
        """
        # Split and normalize.
        print("=" * 10 + " " + "(Prep)rocessing" + " " + "=" * 10)
        (
            self.meta_indices_train, self.meta_indices_valid,
            self.meta_indices_test,
        ) = (
            self.metaset.fitsplit(proportion, priority, self.metaspindle)
            if train_prop[1] == 0 else
            self.metaset.reducesplit(
                proportion, priority, self.metaspindle, *train_prop,
            )
        )
        meta_size_train = len(self.meta_indices_train)
        meta_size_valid = len(self.meta_indices_valid)
        meta_size_test = len(self.meta_indices_test)
        meta_size = meta_size_train + meta_size_valid + meta_size_test
        self.factors = (
            self.metaset.normalizeby(self.meta_indices_train, self.metaspindle)
        )
        # self.factors = None
        print(
            info5(
                {
                    "Split": {
                        "Train": (
                            "{:d}/{:d}".format(meta_size_train, meta_size)
                        ),
                        "Validate": (
                            "{:d}/{:d}".format(meta_size_valid, meta_size)
                        ),
                        "Test": (
                            "{:d}/{:d}".format(meta_size_test, meta_size)
                        ),
                    },
                },
            )
        )
        print(self.metaset.distrep(n=3))

    def fit(
        self,
        proportion: Tuple[int, int, int], priority: Tuple[int, int, int],
        train_prop: Tuple[int, int, bool],
        /,
        *,
        batch_size: int, max_epochs: int, validon: int, validrep: str,
        patience: int,
    ) -> None:
        R"""
        Fit neural network of the framework based on initialization status.
        """
        #
        self.preprocess(proportion, priority, train_prop)

        if self.metaspindle == 'node':
            lb_cnts = [onp.sum(self.metaset.node_labels[self.meta_indices_test] == i) 
                    for i in range(self.metaset.node_labels.max()+1)]
            lb_cnts = onp.array(lb_cnts)
            print(f'label counts testset: {lb_cnts} ({[f"{v:.2f}%" for v in lb_cnts / lb_cnts.sum()]})')
        #
        timecosts: TIMECOST

        # Zero-out number of tuning epochs for non-parametric cases.
        if self.neuralnet.num_resetted_params == 0:
            #
            max_epochs = 0

        #
        rng = onp.random.RandomState(self.seed)
        timecosts = {}
        meta_index_pad = (
            onp.min(self.meta_indices_train).item() if self.BATCH_PAD else None
        )
        epochlen = len(str(max_epochs))
        logs = []
        metric_best = 0.0
        num_not_improving = 0
        noimplen = len(str(patience))

        # Pin shared memory.
        elapsed = time.time()
        pinned_numpy = self.metaset.pin(batch_size)
        timecosts["pin.generate"] = time.time() - elapsed
        elapsed = time.time()
        pinned_ondev = transfer(pinned_numpy, self.device)
        timecosts["pin.transfer"] = time.time() - elapsed

        # Validate once before training.
        print("=" * 10 + " " + "Train & Validate" + " " + "=" * 10)
        with torch.no_grad():
            #
            (metrics, timeparts) = (
                self.evaluate(
                    self.meta_indices_valid.tolist(), meta_index_pad,
                    batch_size, pinned_ondev,
                )
            )
        timecosts["valid.generate"] = timeparts["generate"]
        timecosts["valid.transfer"] = timeparts["transfer"]
        timecosts["valid.forward"] = timeparts["forward"]
        logs.append(metrics)

        # Initialize performance status.
        torch.save(self.neuralnet.state_dict(), self.ptnnp)
        metric_best = metrics[validon]
        num_not_improving = 0
        print(
            "[{:>{:d}d}/{:d}] {:s}: {:>8s} ({:>8s}) {:s}{:<{:d}s}".format(
                0, epochlen, max_epochs, validrep,
                "{:.6f}".format(metrics[validon])[:8],
                "{:.6f}".format(metric_best)[:8],
                "\x1b[92m↑\x1b[0m", "", noimplen,
            ),
        )

        #
        timecosts["train.generate"] = []
        timecosts["train.transfer"] = []
        timecosts["train.forward"] = []
        timecosts["train.backward"] = []
        for epoch in tqdm(range(1, 1 + max_epochs)):
            #
            shuffling = rng.permutation(len(self.meta_indices_train))
            meta_indices_train = self.meta_indices_train[shuffling]

            # Train.
            timeparts = (
                self.train(
                    meta_indices_train.tolist(), meta_index_pad, batch_size,
                    pinned_ondev,
                )
            )
            timecosts["train.generate"] = timeparts["generate"]
            timecosts["train.transfer"] = timeparts["transfer"]
            timecosts["train.forward"] = timeparts["forward"]
            timecosts["train.backward"] = timeparts["backward"]

            # Validate.
            with torch.no_grad():
                #
                (metrics, timeparts) = (
                    self.evaluate(
                        self.meta_indices_valid.tolist(), meta_index_pad,
                        batch_size, pinned_ondev,
                    )
                )
            timecosts["valid.generate"] = timeparts["generate"]
            timecosts["valid.transfer"] = timeparts["transfer"]
            timecosts["valid.forward"] = timeparts["forward"]
            logs.append(metrics)

            # Update performance status
            if (
                metrics[validon] < metric_best - IMP_ABS
                or metrics[validon] < metric_best * (1 - IMP_REL)
            ):
                #
                torch.save(self.neuralnet.state_dict(), self.ptnnp)
                metric_best = metrics[validon]
                improving = True
                num_not_improving = 0
            else:
                #
                improving = False
                num_not_improving = num_not_improving + 1
            print(
                "[{:>{:d}d}/{:d}] {:s}: {:>8s} ({:>8s}) {:s}{:<{:d}s}{:s}"
                .format(
                    epoch, epochlen, max_epochs, validrep,
                    "{:.6f}".format(metrics[validon])[:8],
                    "{:.6f}".format(metric_best)[:8],
                    "\x1b[92m↑\x1b[0m" if improving else "\x1b[91m↓\x1b[0m",
                    "" if improving else str(num_not_improving), noimplen,
                    " --" if num_not_improving == patience else "",
                ),
            )

            # Adjust learning rate according to performance status if
            # necessary.
            if patience >= 0 and num_not_improving == patience:
                # Reduce the learning rate if the neural network is not
                # improving for a while.
                lr_reach_epsilon = False
                if self.neuralnet.num_resetted_params > 0:
                    #
                    for group in self.optim.param_groups:
                        #
                        group["lr"] = group["lr"] / LR_DECAY
                        if group["lr"] < LR_THRES:
                            #
                            lr_reach_epsilon = True
                    num_not_improving = 0
                else:
                    # Directly terminate on the first decay for non-parametric
                    # case.
                    lr_reach_epsilon = True
                if lr_reach_epsilon:
                    # Early stop if learning rate is too small.
                    break
        if hasattr(self.neuralnet, "tgnn"):
            #
            torch.save(
                (
                    torch.cuda.get_device_name(),
                    getattr(getattr(self.neuralnet, "tgnn"), "COSTS"),
                ),
                self.ptres,
            )

        # Final test after training.
        print("=" * 10 + " " + "Test" + " " + "=" * 10)
        with torch.no_grad():
            #
            (metrics_valid, timeparts) = (
                self.evaluate(
                    self.meta_indices_valid.tolist(), meta_index_pad,
                    batch_size, pinned_ondev,
                )
            )
            (metrics_test, timeparts) = (
                self.evaluate(
                    self.meta_indices_test.tolist(), meta_index_pad,
                    batch_size, pinned_ondev,
                )
            )
        timecosts["test.generate"] = timeparts["generate"]
        timecosts["test.transfer"] = timeparts["transfer"]
        timecosts["test.forward"] = timeparts["forward"]
        print(
            "Valid\x1b[94m:\x1b[0m \x1b[3m{:s}\x1b[0m: {:s}"
            .format(validrep, "{:.6f}".format(metrics_valid[validon])[:8]),
        )
        print(
            " Test\x1b[94m:\x1b[0m \x1b[3m{:s}\x1b[0m: {:s}"
            .format(validrep, "{:.6f}".format(metrics_test[validon])[:8]),
        )

        #
        print("=" * 10 + " " + "(Res)ource (Stat)istics" + " " + "=" * 10)
        gpu_mem_peak = int(onp.ceil(torch.cuda.max_memory_allocated() / 1024))
        print("Max GPU Memory: {:d} KB".format(gpu_mem_peak))
        torch.save(
            (
                self.factors, logs, metrics_valid[validon], metrics_test,
                gpu_mem_peak, timecosts
            ),
            self.ptlog,
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