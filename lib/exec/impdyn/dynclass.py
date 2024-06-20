R"""
"""
#
import argparse
import os
import numpy as onp
from typing import List, cast
from ..arguments.dynclass import add_dynclass_arguments
from ..arguments.neuralnet import add_neuralnet_arguments
from ..arguments.tune import add_tune_arguments
from ..utils.floatrep import floatrep
from ...data.dynclass import Brain10, DynCSL, DBLP5, Reddit4
from ..model import encoderize, TIMESTAMPED, NOTEMBEDS, PRETAINABLES
from ...task.impdyn_cls import ImpDynModelCls
from ...task.dyncsl import GraphWindowClassification
from ...task.classification import CE, ERR, MACRO
from ...framework.dyngraph.impdyngraph import FrameworkImplicitDynamicGraph
from ...framework.dyngraph.dyngraphev import FrameworkDynamicGraphForEval
from ...meta.dyngraph.sparse.dynedge import DynamicAdjacencyListDynamicEdge
import torch

def identifier(**KWARGS) -> str:
    R"""
    Get identifier of given arguments.
    """
    #
    res = (
        "{:s}~{:s}~{:s}~{:s}_{:s}~{:s}~{:s}_{:s}~{:s}~{:s}~{:s}_{:s}".format(
            "{:s}~{:s}".format(
                cast(str, KWARGS["source"]), cast(str, KWARGS["train_prop"]),
            )
            if "train_prop" in KWARGS else
            cast(str, KWARGS["source"]),
            cast(str, KWARGS["target"]),
            {"transductive": "trans", "inductive": "induc"}
            [cast(str, KWARGS["framework"])],
            cast(str, KWARGS["win_aggr"]), cast(str, KWARGS["model"]),
            cast(str, KWARGS["hidden"]), cast(str, KWARGS["activate"]),
            cast(str, KWARGS["lr"]), cast(str, KWARGS["weight_decay"]),
            cast(str, KWARGS["clipper"]), cast(str, KWARGS["patience"]),
            cast(str, KWARGS["seed"]),
        )
    )
    args = KWARGS["args"]
    if len(args.exp_name) > 0:
        res += "~" + args.exp_name
    return res


def get_label_counts(
    metaset: DynamicAdjacencyListDynamicEdge, spindle: str,
    /,
) -> List[int]:
    R"""
    Get label counts.
    """
    #
    (train_indices, valid_indices, test_indices) = (
        metaset.fitsplit((7, 1, 2), (2, 1, 0), spindle)
    )
    buf = []
    for indices in (train_indices, valid_indices, test_indices):
        #
        if metaset.node_labels.ndim == 3:
            #
            labels = onp.unique(metaset.node_labels)
            labels = labels[labels >= 0]
            buf.append([1] * len(labels))
        else:
            #
            labels = metaset.node_labels[indices]
            labels = onp.reshape(labels, (len(labels),))
            label_counts = onp.zeros((len(onp.unique(labels)),), dtype=int)
            onp.add.at(label_counts, labels, 1)
            buf.append(cast(List[int], label_counts.tolist()))
    (train_counts, valid_counts, test_counts) = buf
    return train_counts


def main(*ARGS):
    R"""
    Main.
    """
    #
    parser = argparse.ArgumentParser(description="Dynamic Classification")
    add_dynclass_arguments(parser)
    add_neuralnet_arguments(parser)
    add_tune_arguments(parser)
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)
    # Localize arguments.
    source = args.source
    target = args.target
    win_aggr = args.win_aggr

    #
    neuralname = args.model
    hidden = args.hidden
    activate = args.activate
    pretrain_seq_node = args.pretrain_seq_node
    pretrain_seq_edge = args.pretrain_seq_edge

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

    # arguments for IDGNN
    args.kappa = 1.0
    args.phi = torch.nn.ReLU()
    args.regression = False
    args.multi_z = (args.one_z == False)
    args.theta = 0.5 
    args.eta_1 = 1.0
    args.eta_2 = 0.01
    

    # Constant arguments.
    num_batch_graphs = 1

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
    datasetize = {"Brain10": Brain10, "DynCSL": DynCSL, "Reddit4": Reddit4,
                  "DBLP5": DBLP5}[source]
    
    winsize = {"Brain10": 12, "DynCSL": 8, "Reddit4": 10, "DBLP5": 10}[source]
    
    window_future_size = {"Brain10": 0, "DynCSL": 0, "Reddit4": 0, "DBLP5": 0}[source]

    spindle = {"transductive": "node", "inductive": "time"}[frame]
    if target == "all":
        #
        targeton = [0]
    else:
        # UNEXPECT:
        # Classification task has fixed prediction.
        raise NotImplementedError("Classification task has fixed prediction.")

    #
    attach_edge_time: List[str]
    attach_node_time: List[str]

    # Get neural network basement.
    neuralitems = neuralname.split("-")
    if len(neuralitems) > 1:
        #
        if len(neuralitems) > 2 or neuralitems[1] != "Seqed":
            # UNEXPECT:
            # Improper neural network suffix.
            raise NotImplementedError(
                "Improper neural network suffix \"{:s}\" for \"{:s}\"."
                .format(neuralitems[0], "-".join(neuralitems[1:])),
            )
        neuralbase = neuralitems[0]
    else:
        #
        neuralbase = neuralname

    #
    if neuralbase in TIMESTAMPED:
        #
        extend_edge_time = TIMESTAMPED[neuralbase]["edge"]
        extend_node_time = TIMESTAMPED[neuralbase]["node"]
    else:
        #
        extend_edge_time = []
        extend_node_time = []
    attach_edge_time = []
    attach_node_time = []

    # Argument description identifier.
    desc = (
        identifier(
            source=source, target=target, framework=frame, win_aggr=win_aggr,
            model=(
                "{:s}-Seqed".format(neuralname)
                if (
                    neuralname in PRETAINABLES
                    and (
                        len(pretrain_seq_node) > 0
                        or len(pretrain_seq_edge) > 0
                    )
                ) else
                neuralname
            ),
            hidden=str(hidden), activate=activate, lr=floatrep(lr),
            weight_decay=floatrep(weight_decay), clipper=clipper,
            patience=str(patience), seed=str(seed),
            args=args,
        )
    )

    # Prepare COVID dataset.
    print("=" * 10 + " " + "Data & Meta" + " " + "=" * 10)
    dataset = datasetize(os.path.join("src", source))

    # Formalize as future prediction task.
    # Predict 1 future frame (day) by 7 historical frames (past week).
    metaset = (
        dataset.asto_dynamic_adjacency_list_dynamic_edge(
            window_history_size=winsize, window_future_size=window_future_size,
            win_aggr=win_aggr, timestamped_edge_times=extend_edge_time,
            timestamped_node_times=extend_node_time,
            timestamped_edge_feats=attach_edge_time,
            timestamped_node_feats=attach_node_time,
        )
    )
    metaset.inputon(["none", "none", "all", "none"]) # (on_edge_feat, on_edge_label, on_node_feat, on_node_label)
    metaset.targeton(["none", "none", "none", targeton]) # (on_edge_feat, on_edge_label, on_node_feat, on_node_label)
    print(metaset)

    # Prepare EngCOVID model.
    print("=" * 10 + " " + "Model & Task" + " " + "=" * 10)
    neuralnet = (
        (
            GraphWindowClassification
            if source == "DynCSL" else
            ImpDynModelCls
        )(
            encoderize(
                neuralbase, metaset.edge_feat_size,
                1 if source == "DynCSL" else metaset.node_feat_size,
                hidden, hidden, activate,
                dyn_edge=True, dyn_node=True,
                num_nodes=metaset.num_nodes * num_batch_graphs,
                tid_ax=1 if source == "DynCSL" else metaset.node_feat_size,
                window_size=winsize, args=args
            ),
            dataset.num_labels, hidden,
            label_counts=get_label_counts(metaset, spindle), activate=activate,
            notembedon=targeton if neuralbase in NOTEMBEDS else [],
        )
    )

    neuralnet.dynedge()
    neuralnet.initialize(seed)
    if neuralname in PRETAINABLES:
        #
        neuralnet.tgnn.pretrain("node", pretrain_seq_node)
        neuralnet.tgnn.pretrain("edge", pretrain_seq_edge)
    if source == "DynCSL":
        #
        neuralnet.tgnn.SIMPLEST = True
    print(neuralnet)

    # Prepare framework.
    framework = (
        (
            FrameworkDynamicGraphForEval
            if source == "DynCSL" and len(resume_eval) > 0 else
            FrameworkImplicitDynamicGraph
        )(
            desc, metaset, neuralnet,
            lr=lr, weight_decay=weight_decay, seed=seed, device=device,
            metaspindle=spindle, gradclip=clipper, eta_1=args.eta_1, eta_2=args.eta_2
        )
    )
    framework.set_node_batching(True)
    framework.BATCH_PAD = source != "DynCSL"
    if len(resume_eval) == 0:
        #
        framework.fit(
            (7, 1, 2), (2, 1, 0), train_prop_tuple,
            batch_size=num_batch_graphs, max_epochs=max_epochs, validon=MACRO,
            validrep="ROCAUC", patience=patience,
        )
    else:
        #
        framework.besteval(
            (7, 1, 2), (2, 1, 0), train_prop_tuple,
            batch_size=num_batch_graphs, validon=MACRO, validrep="ROCAUC",
            resume=resume_eval,
        )