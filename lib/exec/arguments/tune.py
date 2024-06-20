R"""
"""
#
import argparse


def add_tune_arguments(parser: argparse.ArgumentParser) -> None:
    R"""
    Add neural network tuning related arguments.
    """
    #
    parser.add_argument(
        "--framework",
        type=str, required=True, help="Graph Learning framework name.",
    )
    parser.add_argument(
        "--train-prop",
        type=str, required=False, default="",
        help="Training indices further split numerator and denominator.",
    )
    parser.add_argument(
        "--epoch",
        type=int, required=True, help="Number of maximum tuning epochs."
    )
    parser.add_argument(
        "--lr",
        type=float, required=True, help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float, required=True, help="Weight decay.",
    )
    parser.add_argument(
        "--clipper",
        type=str, required=True, help="Gradient clipper name.",
    )
    parser.add_argument(
        "--patience",
        type=int, required=True,
        help=(
            "Learning rate schedule patience (negative value means constant "
            "learning rate)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int, required=True, help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str, required=True, help="Computing device.",
    )
    parser.add_argument(
        "--resume-eval",
        type=str, required=False, default="",
        help="Evaluation from corresponding log under given directory.",
    )

    parser.add_argument(
            "--one-z",
            action='store_true', help="Use one layer for embedding in implicit model", 
            default=False,
        )
    
    parser.add_argument(
            "--multi-x",
            action='store_true', help="Use multi layers for node features in implicit model", 
            default=False,
        )

    parser.add_argument(
        "--exp-name",
        type=str, required=False, help="Experiment name.", default="",
    )