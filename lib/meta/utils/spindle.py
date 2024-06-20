R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import itertools
import more_itertools as xitertools
from typing import List


def fitsplit(
    indices: onpt.NDArray[onp.generic], proportion: List[int],
    priority: List[int], lab_avail_nodes: List[int] = None,
) -> List[onpt.NDArray[onp.generic]]:
    R"""
    Split given indices fitting to given proportion as close as possible with
    minor adjustment based on priority.
    """
    # Safety check.
    if onp.gcd.reduce(proportion) != 1:
        # EXPECT:
        # It is possible to have improper arguments.
        raise RuntimeError(
            "Proportion of all splits should have greatest common divisor "
            "being 1 to work properly."
        )
    if len(onp.unique(priority)) != len(priority):
        # It is possible to have improper arguments.
        raise RuntimeError(
            "Priority of all splits should be unique to work properly."
        )

    # Break raw indices into chunks fitting the split proportion unit.
    # Get the unit chunk coverage of all splits.
    if lab_avail_nodes is not None:
        indices = lab_avail_nodes

    chunkunits = xitertools.distribute(sum(proportion), indices)
    ties = list(itertools.accumulate(xitertools.prepend(0, proportion)))
    #
    splits: List[List[int]]

    # Fill each split with its covered unit chunks in order of priority.
    splits = [[] for _ in proportion]
    for (sp, begin, end) in (
        zip(
            xitertools.sort_together(
                (priority, range(len(proportion))),
                reverse=True,
            )[1],
            ties[:-1], ties[1:],
        )
    ):
        # Use roundrobin to concatenate sample chains of the same chunk to keep
        # the priority ordering in the chunk.
        splits[sp].extend(xitertools.roundrobin(*chunkunits[begin:end]))
    return [onp.array(split) for split in splits]