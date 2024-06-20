R"""
"""
#
import argparse


def add_slurm_arguments(parser: argparse.ArgumentParser) -> None:
    R"""
    Add Slurm related arguments.
    """
    #
    parser.add_argument(
        "--schedule",
        type=str, required=True, help="Slurm schedule file.",
    )
    parser.add_argument(
        "--nosubmit",
        action="store_true",
        help="Generate shell scripts only without submitting them.",
    )
    parser.add_argument(
        "--only",
        type=str, required=False,
        help="Submit only given queue names (delimited by \",\")",
    )