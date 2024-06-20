R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import more_itertools as xitertools
from typing import Tuple, Union, List
from ..stats.boxlike import BOXLIKE, boxlike as boxlike0


def boxlike5(stats: List[BOXLIKE], /, *, n: int) -> str:
    R"""
    Get visible box-like statistics of given array.
    """
    #
    maxlens = (
        [
            max(len("{:.3f}".format(val)) for val in chunk)
            for chunk in xitertools.divide(7, xitertools.roundrobin(*stats))
        ]
    )
    lines = (
        [
            "[{:>{:d}s} | {:>{:d}s} <-- {:>{:d}s} ({:>{:d}s} ± {:>{:d}s}) "
            "--> {:>{:d}s} | {:>{:d}s}]".format(
                *xitertools.roundrobin(
                    ("{:.3f}".format(val) for val in columns), maxlens,
                ),
            )
            for columns in stats
        ]
    )

    # Skip if there are too many lines.
    # Pay attention that tie1 is open, while tie2 is close.
    tie = len(lines) // 2
    tie1 = min(n, tie + 1)
    tie2 = max(len(lines) - n, tie1)
    skip = [] if tie1 == tie2 else ["..."]
    return "\n".join(lines[:tie1] + skip + lines[tie2:])


def boxlike(
    array: onpt.NDArray[onp.generic],
    /,
    *,
    axis: Union[int, Tuple[int, ...]], n: int,
) -> Tuple[str, List[BOXLIKE]]:
    R"""
    Get visible box-like statistics of given array.
    """
    #
    stats = boxlike0(array, axis=axis)
    string = boxlike5(stats, n=n)
    return (string, stats)