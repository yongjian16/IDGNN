R"""
"""
#
import argparse


def add_dynclass_arguments(parser: argparse.ArgumentParser) -> None:
    R"""
    Add dynamic classification related arguments.
    """
    #
    parser.add_argument(
        "--source",
        type=str, required=True,
        help="Dynamic classification source (dataset).",
    )
    parser.add_argument(
        "--target",
        type=str, required=True, help="Dynamic classification target.",
    )
    parser.add_argument(
        "--win-aggr",
        type=str, required=True, help="Dynamic classification aggregation."
    )