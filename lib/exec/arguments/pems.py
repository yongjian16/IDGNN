R"""
"""
#
import argparse


def add_pems_arguments(parser: argparse.ArgumentParser) -> None:
    R"""
    Add PeMS related arguments.
    """
    #
    parser.add_argument(
        "--source",
        type=str, required=True, help="PeMS source (district).",
    )
    parser.add_argument(
        "--target",
        type=str, required=True, help="PeMS target.",
    )