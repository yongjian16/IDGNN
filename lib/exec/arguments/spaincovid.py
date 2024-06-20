R"""
"""
#
import argparse


def add_spaincovid_arguments(parser: argparse.ArgumentParser) -> None:
    R"""
    Add Spain-COVID related arguments.
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