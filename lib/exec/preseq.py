R"""
"""
#
import argparse
import os
import numpy as onp
import numpy.typing as onpt
from typing import Tuple, List, cast
from .arguments.preseq import add_preseq_arguments
from .arguments.neuralnet import add_neuralnet_arguments
from .arguments.tune import add_tune_arguments
from .utils.floatrep import floatrep
from ..utils.info import info5
from ..meta.meta import Meta
from ..data.pems import PeMS04, PeMS08
from ..data.spaincovid import SpainCOVID
from ..data.engcovid import EngCOVID
from ..meta.vecseq import VectorSequence
from ..task.encdec import SelfEncoderDecoder
from ..task.regression import RMSE
from ..framework.vecseq.vecseq import FrameworkVectorSequence


def identifier(**KWARGS) -> str:
    R"""
    Get identifier of given arguments.
    """
    #
    if "win_aggr" in KWARGS:
        #
        return (
            "{:s}-{:s}~{:s}~{:s}~{:s}_{:s}~{:s}~{:s}_{:s}~{:s}~{:s}~{:s}_{:s}"
            .format(
                "{:s}~{:s}".format(
                    cast(str, KWARGS["source"]),
                    cast(str, KWARGS["train_prop"]),
                )
                if "train_prop" in KWARGS else
                cast(str, KWARGS["source"]),
                cast(str, KWARGS["part"]), cast(str, KWARGS["target"]),
                {"transductive": "trans", "inductive": "induc"}
                [cast(str, KWARGS["framework"])],
                cast(str, KWARGS["win_aggr"]), cast(str, KWARGS["model"]),
                cast(str, KWARGS["hidden"]), cast(str, KWARGS["activate"]),
                cast(str, KWARGS["lr"]), cast(str, KWARGS["weight_decay"]),
                cast(str, KWARGS["clipper"]), cast(str, KWARGS["patience"]),
                cast(str, KWARGS["seed"]),
            )
        )
    else:
        #
        return (
            "{:s}-{:s}~{:s}~{:s}_{:s}~{:s}~{:s}_{:s}~{:s}~{:s}~{:s}_{:s}"
            .format(
                "{:s}~{:s}".format(
                    cast(str, KWARGS["source"]),
                    cast(str, KWARGS["train_prop"]),
                )
                if "train_prop" in KWARGS else
                cast(str, KWARGS["source"]),
                cast(str, KWARGS["part"]), cast(str, KWARGS["target"]),
                {"transductive": "trans", "inductive": "induc"}
                [cast(str, KWARGS["framework"])],
                cast(str, KWARGS["model"]), cast(str, KWARGS["hidden"]),
                cast(str, KWARGS["activate"]), cast(str, KWARGS["lr"]),
                cast(str, KWARGS["weight_decay"]),
                cast(str, KWARGS["clipper"]), cast(str, KWARGS["patience"]),
                cast(str, KWARGS["seed"]),
            )
        )


def prepreprocess(
    metaset: Meta, metaspindle: str,
    proportion: Tuple[int, int, int], priority: Tuple[int, int, int],
    train_prop: Tuple[int, int, bool],
    /,
) -> Tuple[
    onpt.NDArray[onp.generic], onpt.NDArray[onp.generic],
    onpt.NDArray[onp.generic], List[List[Tuple[float, float]]]
]:
    R"""
    Preprocess metaset before fitting.
    """
    # Split and normalize.
    print("=" * 10 + " " + "(Prep)rocessing" + " " + "=" * 10)
    (
        meta_indices_train, meta_indices_valid, meta_indices_test,
    ) = (
        metaset.fitsplit(proportion, priority, metaspindle)
        if train_prop[1] == 0 else
        metaset.reducesplit(proportion, priority, metaspindle, *train_prop)
    )
    meta_size_train = len(meta_indices_train)
    meta_size_valid = len(meta_indices_valid)
    meta_size_test = len(meta_indices_test)
    meta_size = meta_size_train + meta_size_valid + meta_size_test
    factors = metaset.normalizeby(meta_indices_train, metaspindle)
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
    print(metaset.distrep(n=3))
    return (
        meta_indices_train, meta_indices_valid, meta_indices_test, factors,
    )


def main(*ARGS):
    R"""
    Main.
    """
    #
    parser = argparse.ArgumentParser(description="PeMS")
    add_preseq_arguments(parser)
    add_neuralnet_arguments(parser)
    add_tune_arguments(parser)
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)

    # Localize arguments.
    source = args.source
    part = args.part
    target = args.target
    win_aggr = args.win_aggr

    #
    neuralname = args.model
    hidden = args.hidden
    activate = args.activate

    #
    frame = args.framework
    train_prop = args.train_prop
    max_epochs = args.epoch
    lr = args.lr
    weight_decay = args.weight_decay
    clipper = args.clipper
    patience = args.patience
    seed = args.seed
    device = args.device
    resume_eval = args.resume_eval

    # Constant arguments.
    num_batch_seqs = 4096

    #
    if len(train_prop) > 0:
        #
        (train_prop_num_str, train_prop_den_str) = train_prop.split("d")
        train_prop_num_str = train_prop_num_str.replace("n", "-")
        train_prop_num = int(train_prop_num_str)
        train_prop_den = int(train_prop_den_str)
        train_prop_neg = train_prop_num < 0
        train_prop_num = -train_prop_num if train_prop_neg else train_prop_num
    else:
        #
        train_prop_num = 0
        train_prop_den = 0
        train_prop_neg = False
    train_prop_tuple = (train_prop_num, train_prop_den, train_prop_neg)

    # Translate arguments.
    datasetize = (
        {
            "PeMS04": PeMS04, "PeMS08": PeMS08, "SpainCOVID": SpainCOVID,
            "EngCOVID": EngCOVID,
        }[source]
    )
    spindle = {"transductive": "node", "inductive": "time"}[frame]
    if source in ("PeMS04", "PeMS08"):
        #
        if target == "all":
            #
            targeton = [0, 1, 2]
        else:
            #
            targeton = [{"flow": 0, "occupy": 1, "speed": 2}[target]]
        window_history_size = 60 // 5
    elif source in ("SpainCOVID", "EngCOVID"):
        #
        if target == "all":
            #
            targeton = [0]
        else:
            # UNEXPECT:
            # COVID only supports predicting all since it only has one feature.
            raise NotImplementedError("COVID only supports predicting all")
        window_history_size = 7
    else:
        # UNEXPECT:
        # Unknown pretraining task.
        raise NotImplementedError(
            "There is no defined pretraining task on \"{:s}\".".format(source),
        )
    window_future_size = 1

    # Argument description identifier.
    if source in ("EngCOVID",):
        #
        desc = (
            identifier(
                source=source, part=part, target=target, framework=frame,
                model=neuralname, hidden=str(hidden), activate=activate,
                lr=floatrep(lr), weight_decay=floatrep(weight_decay),
                clipper=clipper, patience=str(patience), seed=str(seed),
                win_aggr=win_aggr,
            )
        )
    else:
        #
        desc = (
            identifier(
                source=source, part=part, target=target, framework=frame,
                model=neuralname, hidden=str(hidden), activate=activate,
                lr=floatrep(lr), weight_decay=floatrep(weight_decay),
                clipper=clipper, patience=str(patience), seed=str(seed),
            )
        )

    # Prepare graph dataset.
    print("=" * 10 + " " + "Data & Meta" + " " + "=" * 10)
    if source in ("PeMS04", "PeMS08"):
        #
        dataset = (
            datasetize(
                os.path.join("src", source),
                aug_minutes=True, aug_weekdays=True,
            )
        )
    else:
        #
        dataset = datasetize(os.path.join("src", source))

    # Formalize as future prediction task.
    if source in ("EngCOVID"):
        #
        metaset = (
            dataset.asto_dynamic_adjacency_list_dynamic_edge(
                window_history_size=window_history_size,
                window_future_size=window_future_size, win_aggr=win_aggr,
                timestamped_edge_times=[], timestamped_node_times=[],
                timestamped_edge_feats=[], timestamped_node_feats=[],
            )
        )
    else:
        #
        metaset = (
            dataset.asto_dynamic_adjacency_list_static_edge(
                window_history_size=window_history_size,
                window_future_size=window_future_size,
                timestamped_edge_times=[], timestamped_node_times=[],
                timestamped_edge_feats=[], timestamped_node_feats=[],
            )
        )
    if source in ("SpainCOVID",):
        # Although edge labels are static and does not matter in pretraining,
        # we still explicitly distinguish from non-edge-label cases.
        metaset.inputon(["all", "all", "all", "none"])
    else:
        #
        metaset.inputon(["all", "none", "all", "none"])
    metaset.targeton(["none", "none", targeton, "none"])
    print(metaset)

    # Fetch full batch data to construct vector sequece data for pretraining.
    print("=" * 10 + " " + "Data & Meta" + " " + "=" * 10)
    (
        prepre_indices_train, prepre_indices_valid, prepre_indices_test,
        _,
    ) = prepreprocess(metaset, spindle, (7, 1, 2), (2, 1, 0), train_prop_tuple)
    if spindle == "node":
        #
        time_indices_train = list(range(len(metaset)))
        time_indices_valid = list(range(len(metaset)))
        time_indices_test = list(range(len(metaset)))
        node_indices_train = prepre_indices_train.tolist()
        node_indices_valid = prepre_indices_valid.tolist()
        node_indices_test = prepre_indices_test.tolist()
    else:
        #
        time_indices_train = prepre_indices_train.tolist()
        time_indices_valid = prepre_indices_valid.tolist()
        time_indices_test = prepre_indices_test.tolist()
        node_indices_train = list(range(metaset.num_nodes))
        node_indices_valid = list(range(metaset.num_nodes))
        node_indices_test = list(range(metaset.num_nodes))

    #
    feat_buf_train: List[onpt.NDArray[onp.generic]]
    feat_buf_valid: List[onpt.NDArray[onp.generic]]
    feat_buf_test: List[onpt.NDArray[onp.generic]]
    label_buf_train: List[onpt.NDArray[onp.generic]]
    label_buf_valid: List[onpt.NDArray[onp.generic]]
    label_buf_test: List[onpt.NDArray[onp.generic]]

    #
    feat_buf_train = []
    feat_buf_valid = []
    feat_buf_test = []
    label_buf_train = []
    label_buf_valid = []
    label_buf_test = []
    for (time_indices, node_indices, feat_buf, label_buf) in (
        [
            (
                time_indices_train, node_indices_train, feat_buf_train,
                label_buf_train,
            ),
            (
                time_indices_valid, node_indices_valid, feat_buf_valid,
                label_buf_valid,
            ),
            (
                time_indices_test, node_indices_test, feat_buf_test,
                label_buf_test,
            ),
        ]
    ):
        #
        for t in time_indices:
            #
            (sample_input, _) = metaset[t]
            if source in ("EngCOVID",):
                # We need to differentiate node and edge parts for temporal
                # graph with dynamic edge.
                (
                    _, sample_edge_feat_input, sample_edge_label_input, _, _,
                    sample_node_feat_input, sample_node_label_input, _,
                ) = sample_input
                if part == "edge":
                    #
                    (sample_feat_input, sample_label_input) = (
                        sample_edge_feat_input, sample_edge_label_input,
                    )
                else:
                    #
                    (sample_feat_input, sample_label_input) = (
                        sample_node_feat_input, sample_node_label_input,
                    )
            else:
                #
                (sample_feat_input, sample_label_input, _) = sample_input
            if sample_feat_input.ndim > 0 and part != "edge":
                #
                feat_buf.append(sample_feat_input[node_indices])
            else:
                #
                feat_buf.append(sample_feat_input)
            if sample_label_input.ndim > 0 and part != "edge":
                #
                label_buf.append(sample_label_input[node_indices])
            else:
                #
                label_buf.append(sample_label_input)

    def construct(
        buf: List[onpt.NDArray[onp.generic]],
        /,
    ) -> onpt.NDArray[onp.generic]:
        R"""
        Construct a data tensor.
        """
        #
        if buf[0].ndim > 0:
            #
            return cast(onpt.NDArray[onp.generic], onp.concatenate(buf))
        else:
            #
            return buf[0]

    #
    feats_train = construct(feat_buf_train)
    feats_valid = construct(feat_buf_valid)
    feats_test = construct(feat_buf_test)
    labels_train = construct(label_buf_train)
    labels_valid = construct(label_buf_valid)
    labels_test = construct(label_buf_test)

    #
    premetaset = (
        VectorSequence(
            feats_train, labels_train, feats_valid, labels_valid, feats_test,
            labels_test,
        )
    )
    print(premetaset)

    # Prepare PeMS model.
    print("=" * 10 + " " + "Model & Task" + " " + "=" * 10)
    neuralnet = (
        SelfEncoderDecoder(
            premetaset.feat_size, hidden,
            reduce=neuralname.lower(), activate=activate,
        )
    )
    neuralnet.initialize(seed)
    print(neuralnet)

    # Prepare framework.
    framework = (
        FrameworkVectorSequence(
            desc, premetaset, neuralnet,
            lr=lr, weight_decay=weight_decay, seed=seed, device=device,
            metaspindle=spindle, gradclip=clipper,
        )
    )
    if len(resume_eval) == 0:
        #
        framework.fit(
            (-1, -1, -1), (-1, -1, -1), (0, 0, False),
            batch_size=num_batch_seqs, max_epochs=max_epochs, validon=RMSE,
            validrep="RMSE", patience=patience,
        )
    else:
        #
        framework.besteval(
            (-1, -1, -1), (-1, -1, -1), (0, 0, False),
            batch_size=num_batch_seqs, validon=RMSE, validrep="RMSE",
            resume=resume_eval,
        )