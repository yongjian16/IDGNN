R"""
"""
#
from typing import Dict, List
from ..model.model import Model
from ..model.notembed import MovingAverage
from ..model.notembed import MovingLast
from ..model.gnnx2osnn import GNNx2oSNN
from ..model.snnognnx2 import SNNoGNNx2
from ..model.gx2brnn import Gx2bLSTMLike
from ..model.gx2crnn import Gx2cGRULike
from ..model.evognnx2 import EvoGNNOx2, EvoGNNHx2
from ..model.tgatx2 import TGATx2
from ..model.tgnoptimx2 import TGNOptimx2
from ..model.tgnx2 import TGNx2
from ..model.idgnn import IDGNN

#
staticable = lambda reduce, dynamic: reduce if dynamic else "static"


def mova(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create moving average model.
    """
    #
    return MovingAverage()


def movl(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create moving last model.
    """
    #
    return MovingLast()


def gru_o_gcnx2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create GRUoGCNx2 model.
    """
    return (
        SNNoGNNx2(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="gcn", reduce_edge=staticable("gru", dyn_edge),
            reduce_node="gru", skip=True, activate=act, concat=False,
        )
    )


def gru_o_gcn2x2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create GRUoGCN2x2 model.
    """
    return (
        SNNoGNNx2(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="gcn", reduce_edge=staticable("gru", dyn_edge),
            reduce_node="gru", skip=False, activate=act, concat=True,
        )
    )


def lstm_o_gcnx2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create LSTMoGCNx2 model.
    """
    return (
        SNNoGNNx2(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="gcn", reduce_edge=staticable("lstm", dyn_edge),
            reduce_node="lstm", skip=True, activate=act, concat=False,
        )
    )


def sa_o_gatx2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create SAoGATx2 model.
    """
    return (
        SNNoGNNx2(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="gcn", reduce_edge=staticable("mha", dyn_edge),
            reduce_node="mha", skip=True, activate=act, concat=False,
        )
    )


def gru_o_ginx2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create GRUoGINx2 model.
    """
    return (
        SNNoGNNx2(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="gin", reduce_edge=staticable("gru", dyn_edge),
            reduce_node="gru", skip=True, activate=act, concat=False,
        )
    )


def lstm_o_ginx2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create LSTMoGINx2 model.
    """
    return (
        SNNoGNNx2(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="gin", reduce_edge=staticable("lstm", dyn_edge),
            reduce_node="lstm", skip=True, activate=act, concat=False,
        )
    )


def gcnx2_o_gru(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create GCNx2oGRU model.
    """
    return (
        GNNx2oSNN(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="gcn", reduce="gru", skip=False, activate=act,
            concat=False,
        )
    )


def gcnx2_o_lstm(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create GCNx2oLSTM model.
    """
    return (
        GNNx2oSNN(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="gcn", reduce="lstm", skip=False, activate=act,
            concat=False,
        )
    )


def dysatx2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create DySATx2 model.
    """
    return (
        GNNx2oSNN(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="gat", reduce="mha", skip=False, activate=act,
            concat=False,
        )
    )


def gcrnm2x2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create DCRNNx2 model.
    """
    return (
        Gx2bLSTMLike(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="cheb", skip=False, activate=act, concat=False,
        )
    )


def dcrnnx2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create DCRNNx2 model.
    """
    return (
        Gx2cGRULike(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="cheb", skip=False, activate=act, concat=False,
        )
    )


def evogcnox2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create EvolveGCN-Ox2 model.
    """
    return (
        EvoGNNOx2(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="gcnub", reduce="lstm", skip=False, activate=act,
        )
    )


def evogcnhx2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create EvolveGCN-Hx2 model.
    """
    return (
        EvoGNNHx2(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="gcnub", reduce="gru", skip=False, activate=act,
        )
    )


def tgatx2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create TGATx2 model.
    """
    return (
        TGATx2(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="tgat", skip=False, activate=act,
            feat_timestamp_axis=tid_ax,
        )
    )


def tgnoptimlx2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create TGNOptimx2 (last node feature) model.
    """
    return (
        TGNOptimx2(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="unimp", reduce_mem="gru", reduce_node="linear",
            skip=False, activate=act, feat_timestamp_axis=tid_ax,
        )
    )


def tgnoptimax2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create TGNOptimx2 (aggregated node feature) model.
    """
    return (
        TGNOptimx2(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="unimp", reduce_mem="gru", reduce_node="gru",
            skip=False, activate=act, feat_timestamp_axis=tid_ax,
        )
    )


def tgnx2(
    n_xs_edge: int, n_xs_node: int, n_ys: int, n_hs: int, act: str,
    dyn_edge: bool, dyn_node: bool, n_vs: int, tid_ax: int,
) -> Model:
    R"""
    Create TGNx2 model.
    """
    return (
        TGNx2(
            n_xs_edge, n_xs_node, n_ys, n_hs,
            convolve="unimp", reduce_mem="gru", reduce_node="linear",
            skip=False, activate=act, feat_timestamp_axis=tid_ax,
            num_nodes=n_vs,
        )
    )


#
TIMESTAMPED: Dict[str, Dict[str, List[str]]]


#
NEURALNETS = {
    "MovA": mova,
    "MovL": movl,
    "GRUoGCNx2": gru_o_gcnx2,
    "GRUoGCN2x2": gru_o_gcn2x2,
    "LSTMoGCNx2": lstm_o_gcnx2,
    "SAoGATx2": sa_o_gatx2,
    "GRUoGINx2": gru_o_ginx2,
    "LSTMoGINx2": lstm_o_ginx2,
    "GCNx2oGRU": gcnx2_o_gru,
    "GCNx2oLSTM": gcnx2_o_lstm,
    "DySATx2": dysatx2,
    "GCRNM2x2": gcrnm2x2,
    "DCRNNx2": dcrnnx2,
    "EvoGCNOx2": evogcnox2,
    "EvoGCNHx2": evogcnhx2,
    "TGATx2": tgatx2,
    "TGNOptimLx2": tgnoptimlx2,
    "TGNOptimAx2": tgnoptimax2,
    "TGNx2": tgnx2
}
TIMESTAMPED = {
    "GRUoGCNx2": {"edge": ["inc"], "node": ["inc"]},
    "GRUoGCN2x2": {"edge": ["inc"], "node": ["inc"]},
    "LSTMoGCNx2": {"edge": ["inc"], "node": ["inc"]},
    "SAoGATx2": {"edge": ["inc"], "node": ["inc"]},
    "GRUoGINx2": {"edge": ["inc"], "node": ["inc"]},
    "LSTMoGINx2": {"edge": ["inc"], "node": ["inc"]},
    "TGATx2": {"edge": ["rel"], "node": []},
    "TGNOptimLx2": {"edge": ["inc", "rel"], "node": ["inc", "rel"]},
    "TGNOptimAx2": {"edge": ["inc", "rel"], "node": ["inc", "rel"]},
    "TGNx2": {"edge": ["rel"], "node": []},
}
NOTEMBEDS = ["MovA", "MovL"]
PRETAINABLES = (
    [
        "GRUoGCNx2", "GRUoGCN2x2", "LSTMoGCNx2", "SAoGATx2", "GRUoGINx2",
        "LSTMoGINx2",
    ]
)


def encoderize(
    name: str, feat_input_size_edge: int, feat_input_size_node: int,
    feat_target_size: int, embed_inside_size: int, activation: str,
    /,
    *,
    dyn_edge: bool, dyn_node: bool, num_nodes: int, tid_ax: int,
    window_size: int = None,
    args: List[str] = None
) -> Model:
    R"""
    Get neural network encoder.

    tid_ax: = node_feat_size
    """
    #
    if name == 'IDGNN':
        return IDGNN(
            feat_input_size_edge, feat_input_size_node, feat_target_size,
            embed_inside_size,
            convolve=None, skip=False, activate=activation, 
            num_nodes=num_nodes, window_size=window_size, kappa=args.kappa,
            phi=args.phi, multi_z=args.multi_z, multi_x=args.multi_x,
            regression=args.regression
        )
    
    return (
        NEURALNETS[name](
            feat_input_size_edge, feat_input_size_node, feat_target_size,
            embed_inside_size, activation, dyn_edge, dyn_node, num_nodes,
            tid_ax,
        )
    )