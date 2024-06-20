R"""
"""
#
import subprocess
import re
from typing import Dict, Tuple
from .stask import SPENDING, SRUNNING


#
MAXLEN_JOBID = 8
MAXLEN_NAME = 32
MAXLEN_TIME=12
MAXLEN_STATE = 4
MAXLEN_USER = 8
MAXLEN_PARTITION = 12
SFORMAT = (
    "%.{:d}i %.{:d}j %.{:d}M %.{:d}t %.{:d}u %.{:d}P".format(
        MAXLEN_JOBID, MAXLEN_NAME, MAXLEN_TIME, MAXLEN_STATE, MAXLEN_USER,
        MAXLEN_PARTITION,
    )
)
STATUS = {"PD": SPENDING, "R": SRUNNING}


def squeue() -> Dict[int, Tuple[int, str]]:
    R"""
    Get squeue info of given user.
    """
    #
    user = subprocess.check_output(["whoami"]).decode("utf-8").strip()
    actives = (
        [
            re.split(r"\s+", line.strip())
            for line in (
                re.split(
                    r"\n",
                    subprocess.check_output(
                        ["squeue", "-u", user, "-o", SFORMAT],
                    ).decode("utf-8").strip(),
                )
            )
        ]
    )

    # Safety check.
    if (
        tuple(actives[0])
        != ("JOBID", "NAME", "TIME", "ST", "USER", "PARTITION")
    ):
        # UNEXPECT:
        # Squeue output must be of specific format.
        raise NotImplementedError(
            "Squeue output does not have requiring format.",
        )

    #
    return {int(jid): (STATUS[st], t) for (jid, _, t, st, _, _) in actives[1:]}