R"""
"""
#
import argparse


def add_neuralnet_arguments(parser: argparse.ArgumentParser) -> None:
    R"""
    Add neural network model related arguments.
    """
    #
    parser.add_argument(
        "--model",
        type=str, required=True, help="Model identifier name.",
    )
    parser.add_argument(
        "--hidden",
        type=int, required=True, help="Hidden embedding size.",
    )
    parser.add_argument(
        "--activate",
        type=str, required=True, help="Model activation name.",
    )
    parser.add_argument(
        "--pretrain-seq-node",
        type=str, required=False, default="",
        help="Model pretrained node sequence submodel path.",
    )
    parser.add_argument(
        "--pretrain-seq-edge",
        type=str, required=False, default="",
        help="Model pretrained edge sequence submodel path.",
    )