R"""
"""
#
import json
import itertools
import os
import re
import copy
from typing import List, Tuple, Dict, Callable, Union, cast
from .stask import Task
from .sinfo import HOSTNAME, SERVICES
from ..utils.floatrep import floatrep


def translate(
    title: str, qname: str, key: str, values: Union[List[str], str],
    /,
) -> List[str]:
    R"""
    Translate argument settings of key and values into proper arguments.
    """
    #
    key = key.replace("_", "-")
    if isinstance(values, list):
        #
        if any(not isinstance(valit, str) for valit in values):
            # UNEXPECT:
            # Schedule argument must be a list of strings.
            raise NotImplementedError(
                "Explicit values of schedule argument \"{:s}\" in squeue "
                "\"{:s}:{:s}\" must be a list of strings."
                .format(key, title, qname),
            )
        args = ["--{:s} {:s}".format(key, valit) for valit in values]
    elif isinstance(values, str):
        #
        if values == "?":
            # Optional argument.
            args = ["--" + key, ""]
        else:
            # UNEXPECT:
            # Schedule argument special identifier is unknown.
            raise NotImplementedError(
                "Special schedule argument \"{:s}\" in squeue \"{:s}:{:s}\" "
                "has unknown identifier \"{:s}\"."
                .format(key, title, qname, values),
            )
    else:
        # UNEXPECT:
        # Slurm schedule file has strict format.
        raise NotImplementedError(
            "Slurm schedule queue argument \"{:s}\" in squeue \"{:s}:{:s}\" "
            "has unsupported definition.".format(key, title, qname),
        )
    return args


def queread(
    title: str, qname: str, qdef: Dict[str, Union[List[str], str]],
    fidentifiers: Dict[str, Callable[..., str]],
    /,
    bufdir: str, nosubmit: bool,
) -> List[Task]:
    R"""
    Read a queue schedule.
    """
    # Collect non-argument schedule info.
    # Hostname must be cross-queue info.
    # Partition, number of cpus and gpus, and execution header must be
    # queue-specific info.
    hostnames = cast(str, qdef["@host"])
    partition = cast(str, qdef["@partition"])
    memory = cast(str, qdef["@memory"]) if "@memory" in qdef else None
    cpus = int(cast(str, qdef["@cpus"])) if "@cpus" in qdef else None
    gpus = int(cast(str, qdef["@gpus"])) if "@gpus" in qdef else None
    executable = cast(str, qdef["@execution"])
    qidentifier = cast(str, qdef["@identifier"])

    # Skip squeue not on current host.
    if HOSTNAME not in hostnames:
        #
        print(
            "Squeue \"\x1b[91;4m{:s}:{:s}\x1b[0m\" is not defined on current "
            "host \"{:s}\", and the submission process is "
            "\x1b[91mskipped\x1b[0m.".format(title, qname, HOSTNAME),
        )
        return []
    if not any(partition in service for service in SERVICES.values()):
        # UNEXPECT:
        # Can not submit to unsuppported service partition.
        raise NotImplementedError(
            "Squeue \"{:s}:{:s}\" requires service partition "
            "\"\x1b[91;4m{:s}\x1b[0m\" which is not provided on \"{:s}\"."
            .format(title, qname, partition, HOSTNAME),
        )

    # Translate all arguments into the same format.
    # A key is an argument if and only if its first charater is
    # pure-alphebatical.
    print(
        "Translate arguments of squeue \"\x1b[94m{:s}\x1b[0m\"."
        .format(qname),
    )
    formatted = []
    for key in filter(lambda key: key[0].isalpha(), qdef.keys()):
        #
        formatted.append(translate(title, qname, key, qdef[key]))

    # Product formatted arguments, thus each item of producted list is full
    # argument configuration of a task.
    print("Product translated arguments of squeue \"{:s}\".".format(qname))
    args = (
        [
            list(filter(lambda config: len(config) > 0, configs))
            for configs in itertools.product(*formatted)
        ]
    )

    # Generate all tasks with full arguments and submission configurations.
    tasks = []
    for taskit in args:
        # Get task identifier from full argument configurations.
        config = {}
        for argit in taskit:
            #
            argeles = argit.split()
            if len(argeles) == 2:
                #
                (argkey, argval) = argeles
                if (
                    re.match("[0-9]+\\.[0-9]+", argval)
                    or re.match("[0-9]+(\\.[0-9]+)?[eE][+-]?[0-9]+", argval)
                ):
                    # Float value need special representation in identifier.
                    argval = floatrep(float(argval))
                config[argkey[2:].replace("-", "_")] = argval
        if len(qidentifier) > 4 and "pre-" == qidentifier[:4]:
            #
            identifier = fidentifiers["preseq"](**config)
        else:
            #
            identifier = fidentifiers[qidentifier](**config)

        # Generate file system related configurations.
        shpath = os.path.join(bufdir, title, qname, identifier + ".sh")
        stdout = (
            os.path.join(bufdir, title, qname, identifier + ".stdout.txt")
        )
        stderr = (
            os.path.join(bufdir, title, qname, identifier + ".stderr.txt")
        )
        tasks.append(
            Task(
                executable, taskit, shpath, stdout, stderr,
                identifier=identifier, partition=partition, memory=memory,
                n_cpus=cpus, n_gpus=gpus, nosubmit=nosubmit,
            )
        )
    return tasks


def schedule(
    path: str, fidentifiers: Dict[str, Callable[..., str]],
    /,
    *,
    bufdir: str, nosubmit: bool, only: List[str],
) -> List[Tuple[str, List[Task]]]:
    R"""
    Generate a schedule list of arguments.
    """
    #
    print("Load json schedule \"\x1b[94m{:s}\x1b[0m\".".format(path))
    with open(path, "r") as file:
        #
        definition = json.load(file)
    title = ".".join(os.path.basename(path).split(".")[:-1])

    # Collect schedule queue names.
    qnames = (
        [
            name[1:]
            for name in filter(lambda key: key[0] == "$", definition.keys())
        ]
    )
    if (
        any(
            len(name) < 8 or name[:6] != "squeue" or not name[6] == "-"
            for name in qnames
        )
    ):
        # UNEXPECT:
        # Slurm schedule file has strict format.
        raise NotImplementedError(
            "Slurm schedule queue names must be defined by \"$squeue-.+\" "
            "format.",
        )
    qnames = [qname[7:] for qname in qnames]
    if len(only) > 0:
        #
        print("Trim submitting queues according to \"only\" argument.")
        qnames = list(set(qnames) & set(only))

    # Collect full argument definitions from all queues.
    queues = []
    for qname in qnames:
        # Get queue definition.
        qdef = copy.deepcopy(definition["$squeue-{:s}".format(qname)])
        if "*" in definition:
            # Fill queue-sharing values.
            qdef.update(definition["*"])
        for key in qdef.keys():
            # Replace 1-level link (integer) by corresponding values.
            if isinstance(qdef[key], int):
                #
                qdef[key] = definition[key + str(qdef[key])]
        queues.append(
            (
                qname,
                queread(
                    title, qname, qdef, fidentifiers,
                    bufdir=bufdir, nosubmit=nosubmit,
                ),
            ),
        )
    return queues