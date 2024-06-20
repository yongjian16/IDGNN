R"""
"""
#
import os
import subprocess
import more_itertools as xitertools
from typing import List, Optional


# Status constants.
SMEMORY = -1
SCRIPTED = 0
SUBMITTED = 1
SPENDING = 2
SRUNNING = 3
SDONE = 4
STATUS = {
    -1: "In-Memory",
    0: "Scripted",
    1: "Submitted",
    2: "Pending",
    3: "Running",
    4: "Done",
}


#
SMAXLEN = 79
SINDENT = 4


class Task(object):
    R"""
    A sbatch task.
    """
    def __init__(
        self,
        executable: str, args: List[str], shpath: str, stdout: str,
        stderr: str,
        /,
        identifier: str, partition: str, memory: Optional[str],
        n_cpus: Optional[int], n_gpus: Optional[int], nosubmit: bool,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        self.executable = executable
        self.args = args
        self.shpath = shpath
        self.identifier = identifier
        self.stdout = stdout
        self.stderr = stderr
        self.partition = partition
        self.memory = memory
        self.n_cpus = n_cpus
        self.n_gpus = n_gpus
        self.nosubmit = nosubmit

        #
        self.status = SMEMORY
        self.user = "(null)"
        self.jobid = -1
        self.n_days = -1
        self.n_hours = -1
        self.n_minutes = -1
        self.n_seconds = -1

    def scomment(self, /) -> List[str]:
        R"""
        Generate sbatch comment block lines.
        """
        # Essential comments.
        lines = []
        lines.append("--job-name={:s}".format(self.identifier))
        lines.append("--output={:s}".format(self.stdout))
        lines.append("--error={:s}".format(self.stderr))
        lines.append("--partition={:s}".format(self.partition))

        # Optional comments.
        if self.memory is not None:
            #
            lines.append("--mem={:s}".format(self.memory))
        if self.n_cpus is not None:
            #
            lines.append("--cpus-per-task={:d}".format(self.n_cpus))
        if self.n_gpus is not None:
            #
            lines.append("--gres=gpu:{:d}".format(self.n_gpus))

        #
        return [" ".join(("#SBATCH", soption)) for soption in lines]

    def sexecutable(self, /) -> List[str]:
        R"""
        Generate all sbatch execution command lines for submission.
        """
        # Task will always be resource-trackable.
        restracker = (
            "/usr/bin/time -f \"Max CPU Memory: %M KB\\nElapsed: %e sec\""
        )

        # Executable bash.
        lines = ["#!/bin/bash"]

        # Add sbatch configuration comments.
        lines.append("")
        lines.extend(self.scomment())

        # Add wrapped executuable header.
        lines.append("")
        lines.append(restracker + " \\")
        lines.append(self.executable)

        #
        arglines: List[str]

        # Execution arguments with max line characters.
        arglines = []
        identation = " " * SINDENT
        for argument in self.args:
            #
            if len(arglines) == 0:
                #
                arglines.append(identation + argument)
            elif len(arglines[-1]) + 1 + len(self.args) > SMAXLEN:
                #
                arglines[-1] = arglines[-1] + " \\"
                arglines.append(identation + argument)
            else:
                #
                arglines[-1] = arglines[-1] + " " + argument
        if len(arglines) > 0:
            #
            lines[-1] = lines[-1] + " \\"
            lines.extend(arglines)
        return lines

    def submit(self, /) -> int:
        R"""
        Submit task and get its job ID.
        """
        #
        if self.status >= SUBMITTED:
            # UNEXPECT:
            # Can not submit multiple times.
            raise NotImplementedError("Submit a submitted task.")

        # Ensure directories.
        for path in (self.shpath, self.stdout, self.stderr):
            #
            directory = os.path.dirname(path)
            if not os.path.isdir(directory):
                # UNEXPECT:
                # Shell script buffer directory must exist.
                raise NotImplementedError(
                    "Submission buffer directory \"{:s}\" does not exist."
                    .format(directory),
                )

            #
            directory = os.path.join(directory, "jobs")
            if not os.path.isdir(directory):
                # Create a directory to hold soft links named by job IDs.
                os.makedirs(directory, exist_ok=True)

        # Ensure sanity of task input and output.
        if not os.path.isfile(self.shpath):
            # UNEXPECT:
            # Shell script must be saved in a file before submission.
            raise NotImplementedError(
                "Submission file \"{:s}\" does not exist.".format(self.shpath),
            )
        if os.path.isfile(self.stdout):
            #
            os.remove(self.stdout)
        if os.path.isfile(self.stderr):
            #
            os.remove(self.stderr)

        # Submit.
        self.user = subprocess.check_output(["whoami"]).decode("utf-8").strip()
        if self.nosubmit:
            # Return negative job ID as pseduo submission if submission is
            # disabled.
            self.status = SUBMITTED
            self.jobid = -1
        else:
            #
            try:
                #
                submitinfo = (
                    subprocess.check_output(["sbatch", self.shpath])
                    .decode("utf-8").strip()
                )
                (_, _, _, jidstr) = submitinfo.split()
            except Exception:
                # Cancel all submitted jobs if we encounter any error.
                subprocess.check_output(["scancel", "-u", self.user])

                # EXPECT:
                # It is possible that submission is rejected, for exmaple,
                # running out of maximum submission.
                raise RuntimeError(
                    "Encounter error on submitting \"{:s}\"."
                    .format(self.shpath),
                )
            self.jobid = int(jidstr)
            self.status = SUBMITTED

        #
        for (pathabs, ext) in (
            (self.shpath, "sh"), (self.stdout, "stdout"),
            (self.stderr, "stderr"),
        ):
            #
            pathrel = (
                os.path.join(
                    os.path.dirname(pathabs), "jobs",
                    "{:d}.{:s}".format(self.jobid, ext),
                )
            )
            if self.jobid >= 0:
                #
                os.symlink(
                    os.path.join("..", os.path.basename(pathabs)), pathrel,
                )
        return self.jobid

    def jupdate(self, status: int, time: str) -> None:
        R"""
        Update job status.
        """
        #
        self.status = status

        #
        (inaday, day) = xitertools.padded(reversed(time.split("-")), "0", 2)
        (second, minute, hour) = (
            xitertools.padded(reversed(inaday.split(":")), "00", 3)
        )
        self.n_days = int(day)
        self.n_hours = int(hour)
        self.n_minutes = int(minute)
        self.n_seconds = int(second)

    def __repr__(self, /) -> str:
        R"""
        Get representation of the class.
        """
        #
        return (
            "[\x1b[91;4m{:d}\x1b[0m]\x1b[92;3m{:s}\x1b[0m/\x1b[92m{:s}\x1b[0m@"
            "\x1b[94m{:s}\x1b[0m(\x1b[93m{:s}\x1b[0m)#\x1b[96m{:d}-{:02d}:"
            "{:02d}:{:02d}\x1b[0m".format(
                self.jobid, self.user, self.identifier, self.partition,
                STATUS[self.status], self.n_days, self.n_hours, self.n_minutes,
                self.n_seconds,
            )
        )