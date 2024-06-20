R"""
"""
#
import numpy as onp
import numpy.typing as onpt
import scipy.sparse as osparse
from typing import List, Callable, Tuple, cast, Optional, Union
from .dynedge import DynamicAdjacencyListDynamicEdge


class DynamicAdjacencyListDynamicEdgeSingle(DynamicAdjacencyListDynamicEdge):
    R"""
    A special children where we only have one window in the dataset.
    """