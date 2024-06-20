R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import more_itertools as xitertools
from typing import Tuple


def bins0(
    degrees: onpt.NDArray[onp.generic],
    /,
    *,
    num: int,
) -> Tuple[onpt.NDArray[onp.generic], onpt.NDArray[onp.generic]]:
    R"""
    Get statistics bins of node degrees.
    """
    #
    degree_min = int(onp.min(degrees))
    degree_max = int(onp.max(degrees))
    degree_size = degree_max - degree_min + 1

    # Degree Maximum is increased by 1 in bin bound generation to ensure that
    # left-close right-open range also applies on the last bin.
    degree_bounds = (
        onp.linspace(
            degree_min, degree_max + 1,
            num=min(degree_size + 1, num), endpoint=True,
        )
    )

    # We focus on percentage rather than count in each bin.
    # Manually normaize rather than use density since numpy density need extra
    # processing to get percentage from density.
    (degree_bins, _) = onp.histogram(degrees, degree_bounds)
    degree_bins = degree_bins / len(degrees)
    return (degree_bounds, degree_bins)


def bins5(
    degree_bounds: onpt.NDArray[onp.generic],
    degree_bins: onpt.NDArray[onp.generic],
    /,
    *,
    num_bins_per_line: int,
) -> str:
    R"""
    Get visible statistics bins of node degrees.
    """
    #
    num_bins_per_line = min(len(degree_bins), num_bins_per_line)
    maxlen = len("{:.1f}".format(onp.max(degree_bounds)))

    #
    return (
        ",\n".join(
            ", ".join(
                "[{:>{mxl:d}s}, {:>{mxl:d}s}): {:>5s}%".format(
                    "{:.1f}".format(lower), "{:.1f}".format(upper),
                    "{:.1f}".format(percent * 100),
                    mxl=maxlen,
                )
                for lower, upper, percent in chunk
            )
            for chunk in (
                xitertools.chunked(
                    zip(
                        xitertools.islice_extended(
                            degree_bounds, None, -1, None,
                        ),
                        xitertools.islice_extended(
                            degree_bounds, 1, None, None,
                        ),
                        degree_bins,
                    ),
                    num_bins_per_line,
                )
            )
        )
    )


def bins(
    degrees: onpt.NDArray[onp.generic],
    /,
    *,
    num_bins: int, num_bins_per_line: int,
) -> Tuple[str, Tuple[onpt.NDArray[onp.generic], onpt.NDArray[onp.generic]]]:
    R"""
    Get visible statistics bins of node degrees.
    """
    #
    (degree_bounds, degree_bins) = bins0(degrees, num=num_bins)
    string = (
        bins5(
            degree_bounds, degree_bins,
            num_bins_per_line=num_bins_per_line,
        )
    )
    return (string, (degree_bounds, degree_bins))