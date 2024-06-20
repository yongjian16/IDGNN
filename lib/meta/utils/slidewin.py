R"""
"""
#
from typing import Optional, Tuple


def sliding_window(
    window_history_size: Optional[int], window_future_size: Optional[int],
    max_size: int,
) -> Tuple[int, int, int]:
    R"""
    Get formalized window sizes.
    """
    # History window size being None means treat the whole time scope as a
    # window.
    # Future window size being None means treat history window also as
    # future window.
    window_history_size = (
        min(max_size, window_history_size)
        if window_history_size is not None else
        max_size
    )
    window_future_size = (
        window_future_size
        if window_future_size is not None else
        -window_history_size
    )
    window_size = window_history_size + max(window_future_size, 0)
    return (window_size, window_history_size, window_future_size)