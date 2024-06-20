R"""
"""
#
import argparse
import os
from ..arguments.slurm import add_slurm_arguments
from .schedule import schedule
from .submit import submit
from ..pems import identifier as pems
from ..spaincovid import identifier as spaincovid
from ..engcovid import identifier as engcovid
from ..dynclass import identifier as dynclass
from ..preseq import identifier as preseq


def main(*ARGS) -> None:
    R"""
    Main.
    """
    #
    parser = argparse.ArgumentParser(description="Slurm Scheduler")
    add_slurm_arguments(parser)
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)

    #
    path = args.schedule
    nosubmit = args.nosubmit
    only = "" if args.only is None else args.only

    # Format schedule.
    fidentifiers = (
        {
            "pems04": pems, "pems08": pems, "spaincovid": spaincovid,
            "engcovid": engcovid, "reddit4": dynclass, "dblp5": dynclass,
            "brain10": dynclass, "dyncsl": dynclass, "preseq": preseq,
        }
    )
    submit(
        ".sbatch", ".".join(os.path.basename(path).split(".")[:-1]),
        schedule(
            path, fidentifiers,
            bufdir=".sbatch", nosubmit=nosubmit,
            only=[qname for qname in only.split(",") if len(qname) > 0],
        ),
    )