R"""
"""
#
import os
import re
import shutil
import numpy as onp
import more_itertools as xitertools
from typing import List, Tuple, Dict
from .stask import Task, SDONE, SMEMORY, SCRIPTED
from .squeue import squeue


#
MAXSUBS = 2048
SUBSTEPS = 10


def queuing(tracker: str) -> bool:
    R"""
    Check if any tasks in tracker is in the host queue.
    """
    # File-not-exist corner case.
    if not os.path.isfile(tracker):
        #
        return False

    # Load all job IDs in tracking file.
    jobids = set([])
    with open(tracker, "r") as file:
        #
        for line in file:
            # Load only active job IDs.
            (_, jidstr, _) = re.split(r"\s+", line.strip())
            jid = int(jidstr)
            if jid >= 0:
                #
                jobids.add(jid)

    # Get queuing job IDs from squeue.
    queids = set(squeue().keys())
    return len(jobids & queids) > 0


def trackout(tracker: str, qtasks: List[Task], /) -> None:
    R"""
    Output task status to tracking file.
    """
    #
    infos = []
    for task in qtasks:
        #
        infos.append((task.identifier, str(task.jobid), str(task.status)))
    maxlens = [max(len(it) for it in col) for col in xitertools.unzip(infos)]

    #
    with open(tracker, "w") as file:
        #
        for info in infos:
            #
            file.write(
                "{:>{:d}s} {:>{:d}s} {:>{:d}s}\n"
                .format(*xitertools.interleave(info, maxlens)),
            )


def generate(root: str, title: str, qname: str, qtasks: List[Task], /) -> None:
    R"""
    Generate all task scripts in a queue.
    """
    #
    tracker = os.path.join(root, "{:s}-{:s}txt".format(title, qname))

    # If any tracked job is still queuing, overwriting is not permitted.
    # Otherwise, we will preserve the queue tracker.
    if queuing(tracker):
        # EXPECT:
        # New tracking file can not collide with tracking file for running
        # jobs.
        raise RuntimeError(
            "At least one job in colliding tracker file \"{:s}\" is still "
            "running.".format(tracker),
        )
    else:
        # Pin tracker file to work exclusively.
        trackout(tracker, qtasks)

    # Ensure submission buffer.
    qdir = os.path.join(root, title, qname)
    if os.path.isdir(qdir):
        #
        shutil.rmtree(qdir)
    os.makedirs(qdir, exist_ok=True)

    #
    stepsize = int(onp.ceil(float(len(qtasks)) / float(SUBSTEPS)))
    for i, task in enumerate(qtasks):
        # Show progress and next workload.
        if i % stepsize == 0:
            #
            tstep = i // stepsize
            tmin = tstep * stepsize + 1
            tmax = min(tmin + stepsize, len(qtasks))
            print(
                "Create submission files `[\x1b[92;4m{:d}\x1b[0m, "
                "\x1b[32m{:d}\x1b[0m]/{:d}` of squeue "
                "\"\x1b[94;4m{:s}:{:s}\x1b[0m\"."
                .format(tmin, tmax, len(qtasks), title, qname),
            )

        #
        with open(task.shpath, "w") as file:
            #
            file.write("\n".join(task.sexecutable()))
        task.status = SCRIPTED
    trackout(tracker, qtasks)


def clear(
    root: str, title: str, queues: List[Tuple[str, List[Task]]],
    /,
) -> None:
    R"""
    Clear buffer for all queues.
    """
    # Ensure all queues are not colliding.
    directory = os.path.join(root, title)
    for (qname, qtasks) in queues:
        #
        tracker = os.path.join(root, title, "{:s}.txt".format(qname))

        # If any tracked job is still queuing, overwriting is not permitted.
        # Otherwise, we will preserve the queue tracker.
        if queuing(tracker):
            # EXPECT:
            # New tracking file can not collide with tracking file for running
            # jobs.
            raise RuntimeError(
                "At least one job in colliding tracker file \"{:s}\" is still "
                "running.".format(tracker),
            )

    # Remove the top tracking directory to clear everything.
    if os.path.isdir(directory):
        #
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

    # Pin tracker files and submission buffer for all queues to work
    # exclusively.
    for (qname, qtasks) in queues:
        #
        tracker = os.path.join(root, title, "{:s}.txt".format(qname))
        trackout(tracker, qtasks)

        # Ensure submission buffer for each queue.
        qdir = os.path.join(directory, qname)
        os.makedirs(qdir, exist_ok=True)


def submit(
    root: str, title: str, queues: List[Tuple[str, List[Task]]],
    /,
) -> None:
    R"""
    Submit all queues.
    """
    # Clear buffer.
    clear(root, title, queues)

    # Ensure submission is within limitation.
    maxsize = sum(len(qtasks) for (_, qtasks) in queues)
    if maxsize > MAXSUBS:
        #
        print(
            "Can not scehdule and submit too many files ({:d}) in the same "
            "time.".format(maxsize),
        )

    # Submit tasks of each queue in order.
    for (qname, qtasks) in queues:
        #
        stepsize = int(onp.ceil(float(len(qtasks)) / float(SUBSTEPS)))
        for i, task in enumerate(qtasks):
            # Show progress and next workload.
            if i % stepsize == 0:
                #
                tstep = i // stepsize
                tmin = tstep * stepsize + 1
                tmax = min(tmin + stepsize, len(qtasks))
                print(
                    "Create submission files `[\x1b[92;4m{:d}\x1b[0m, "
                    "\x1b[32m{:d}\x1b[0m]/{:d}` of squeue "
                    "\"\x1b[94;4m{:s}:{:s}\x1b[0m\"."
                    .format(tmin, tmax, len(qtasks), title, qname),
                )

            # Save task script.
            with open(task.shpath, "w") as file:
                #
                file.write("\n".join(task.sexecutable()))
            task.status = SCRIPTED

        # Update tracker file.
        tracker = os.path.join(root, title, "{:s}.txt".format(qname))
        trackout(tracker, qtasks)

    #
    for (qname, qtasks) in queues:
        # Submit all tasks.
        for task in qtasks:
            #
            task.submit()

        # Update tracker file.
        tracker = os.path.join(root, title, "{:s}.txt".format(qname))
        trackout(tracker, qtasks)