R"""
"""
#
import numpy as onp
import numpy.typing as onpt
from typing import Tuple


#
EDGEONLY = (
    Tuple[
        onpt.NDArray[onp.generic], onpt.NDArray[onp.generic],
        onpt.NDArray[onp.generic], onpt.NDArray[onp.generic],
    ]
)
NODEONLY = Tuple[onpt.NDArray[onp.generic], onpt.NDArray[onp.generic]]
GRAPHONLY = Tuple[onpt.NDArray[onp.generic], onpt.NDArray[onp.generic]]