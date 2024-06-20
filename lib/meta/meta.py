R"""
"""
#
import abc
import numpy as onp
import numpy.typing as onpt
from typing import List, Tuple, Union
from .types import MEMPIN, MEMBAT
from ..utils.info import INFO, info5


class Meta(abc.ABC):
    R"""
    Formatted metaset.
    This is used to enforce arbitrary dataset into specific format for the
    consistency requirement in later usage.
    """
    @abc.abstractmethod
    def shapize(self, /) -> None:
        R"""
        Get data shapes.
        """
        #
        ...

    @abc.abstractmethod
    def collect(self, /) -> None:
        R"""
        Collect essential statistics.
        """
        #
        ...

    def repr(self, /) -> INFO:
        R"""
        Get essential statistics for representation.
        """
        #
        return {"Basic": {"#Samples": str(len(self))}}

    def __repr__(self, /) -> str:
        R"""
        Get representaiton of the class.
        """
        #
        return info5(self.repr())

    @abc.abstractmethod
    def fitsplit(
        self,
        proportion: Tuple[int, int, int], priority: Tuple[int, int, int],
        dim: str,
        /,
    ) -> Tuple[
        onpt.NDArray[onp.generic], onpt.NDArray[onp.generic],
        onpt.NDArray[onp.generic],
    ]:
        R"""
        Split data indices into training, validation and test indices along
        dimension identifier by given name for fitting.
        """
        #
        ...

    def reducesplit(
        self,
        proportion: Tuple[int, int, int], priority: Tuple[int, int, int],
        dim: str, prop_keep: int, prop_total: int, inverse: bool,
        /,
    ) -> Tuple[
        onpt.NDArray[onp.generic], onpt.NDArray[onp.generic],
        onpt.NDArray[onp.generic],
    ]:
        R"""
        Split data indices into training, validation and test indices along
        dimension identifier by given name.
        """
        # UNEXPECT:
        # Should not be used except for special cases.
        raise RuntimeError("Should not reduce data except for special cases.")

    @abc.abstractmethod
    def normalizeby(
        self,
        indices: onpt.NDArray[onp.generic], spindle: str,
        /,
    ) -> List[List[Tuple[float, float]]]:
        R"""
        Normalize using statistics only from given indices.
        """
        #
        ...

    @abc.abstractmethod
    def inputon(self, on: List[Union[str, List[int]]]) -> None:
        R"""
        Set input feature and label data columns.
        """
        #
        ...

    @abc.abstractmethod
    def targeton(self, on: List[Union[str, List[int]]]) -> None:
        R"""
        Set tatget feature and label data columns.
        """
        #
        ...

    @abc.abstractmethod
    def distrep(self, /, *, n: int) -> str:
        R"""
        Get data distribution representation of the class.
        """
        #
        ...

    @abc.abstractmethod
    def pin(self, batch_size: int, /) -> MEMPIN:
        R"""
        Always-shared data to be pinned into device memory.
        Shared data should not differentiate input and target.
        """
        #
        ...

    @abc.abstractmethod
    def __len__(self, /) -> int:
        R"""
        Length of the class.
        """
        #
        ...

    @abc.abstractmethod
    def __getitem__(self, idx: int, /) -> MEMBAT:
        R"""
        Get an indexable item of the class.
        """
        #
        ...