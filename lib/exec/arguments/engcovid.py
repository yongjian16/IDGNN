R"""
"""
#
import argparse


def add_engcovid_arguments(parser: argparse.ArgumentParser) -> None:
    R"""
    Add Eng-COVID related arguments.
    """
    #
    parser.add_argument(
        "--source",
        type=str, required=True, help="COVID source (country).",
    )
    parser.add_argument(
        "--target",
        type=str, required=True, help="COVID target.",
    )
    parser.add_argument(
        "--win-aggr",
        type=str, required=True, help="COVID window aggregation."
    )