R"""
"""
#
import argparse


def add_preseq_arguments(parser: argparse.ArgumentParser) -> None:
    R"""
    Add pretraining sequence related arguments.
    """
    #
    parser.add_argument(
        "--source",
        type=str, required=True, help="Pretraining source graph dataset.",
    )
    parser.add_argument(
        "--part",
        type=str, required=True,
        help="Pretraining source part (node or edge).",
    )
    parser.add_argument(
        "--target",
        type=str, required=True, help="Pretraining target.",
    )
    parser.add_argument(
        "--win-aggr",
        type=str, required=False, default="",
        help="Pretraining window aggregation (if exists).",
    )