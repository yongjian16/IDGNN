R"""
"""
#
import argparse
import os
from typing import List, cast
from .arguments.engcovid import add_engcovid_arguments
from .arguments.neuralnet import add_neuralnet_arguments
from .arguments.tune import add_tune_arguments
from .utils.floatrep import floatrep
from ..data.engcovid import EngCOVID
from .model import encoderize, TIMESTAMPED, NOTEMBEDS, PRETAINABLES
from ..task.disspread import DiseaseSpread
from ..task.regression import RMSE
from ..framework.dyngraph.dyngraph import FrameworkDynamicGraph


def identifier(**KWARGS) -> str:
    R"""
    Get identifier of given arguments.
    """
    #
    return (
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


def main(*ARGS):
    R"""
    Main.
    """
    #
    parser = argparse.ArgumentParser(description="EngCOVID")
    add_engcovid_arguments(parser)
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

    # Constant arguments.
    # TODO: Rerun by 64 later (which should not have great difference).
    num_batch_graphs = 4

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
    datasetize = {"EngCOVID": EngCOVID}[source]
    elsize = {"EngCOVID": cast(List[int], [])}[source]
    spindle = {"transductive": "node", "inductive": "time"}[frame]
    if target == "all":
        #
        targeton = [0]
    else:
        # UNEXPECT:
        # COVID only supports predicting all since it only has one feature.
        raise NotImplementedError("EngCOVID only supports predicting all")

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
        )
    )

    # Prepare COVID dataset.
    print("=" * 10 + " " + "Data & Meta" + " " + "=" * 10)
    dataset = datasetize(os.path.join("src", source))

    # Formalize as future prediction task.
    # Predict 1 future frame (day) by 7 historical frames (past week).
    metaset = (
        dataset.asto_dynamic_adjacency_list_dynamic_edge(
            window_history_size=7, window_future_size=1, win_aggr=win_aggr,
            timestamped_edge_times=extend_edge_time,
            timestamped_node_times=extend_node_time,
            timestamped_edge_feats=attach_edge_time,
            timestamped_node_feats=attach_node_time,
        )
    )
    metaset.inputon(["all", "none", "all", "none"]) # (on_edge_feat, on_edge_label, on_node_feat, on_node_label)
    metaset.targeton(["none", "none", targeton, "none"]) # (on_edge_feat, on_edge_label, on_node_feat, on_node_label)
    print(metaset)

    # Prepare EngCOVID model.
    print("=" * 10 + " " + "Model & Task" + " " + "=" * 10)
    neuralnet = (
        DiseaseSpread(
            encoderize(
                neuralbase, metaset.edge_feat_size, metaset.node_feat_size,
                hidden, hidden, activate,
                dyn_edge=True, dyn_node=True,
                num_nodes=metaset.num_nodes * num_batch_graphs,
                tid_ax=metaset.node_feat_size,
            ),
            len(targeton), hidden,
            edge_num_labels=elsize, activate=activate,
            notembedon=targeton if neuralbase in NOTEMBEDS else [],
        )
    )
    neuralnet.dynedge()
    neuralnet.initialize(seed)
    if neuralname in PRETAINABLES:
        #
        neuralnet.tgnn.pretrain("node", pretrain_seq_node)
        neuralnet.tgnn.pretrain("edge", pretrain_seq_edge)
    print(neuralnet)

    # Prepare framework.
    framework = (
        FrameworkDynamicGraph(
            desc, metaset, neuralnet,
            lr=lr, weight_decay=weight_decay, seed=seed, device=device,
            metaspindle=spindle, gradclip=clipper,
        )
    )
    framework.set_node_batching(True)
    if len(resume_eval) == 0:
        #
        framework.fit(
            (7, 1, 2), (2, 1, 0), train_prop_tuple,
            batch_size=num_batch_graphs, max_epochs=max_epochs, validon=RMSE,
            validrep="RMSE", patience=patience,
        )
    else:
        #
        framework.besteval(
            (7, 1, 2), (2, 1, 0), train_prop_tuple,
            batch_size=num_batch_graphs, validon=RMSE, validrep="RMSE",
            resume=resume_eval,
        )