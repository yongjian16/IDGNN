R"""
"""
#
import numpy as onp
import numpy.typing as onpt
from typing import Tuple, List, Union


#
MEMOBJ = onpt.NDArray[onp.generic]
MEMPIN = List[MEMOBJ]
MEMBAT = Tuple[List[MEMOBJ], List[MEMOBJ]]