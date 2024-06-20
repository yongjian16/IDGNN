R"""
"""
#
import numpy as onp
import numpy.typing as onpt
from typing import Union, List, Tuple
from .meta import Meta
from ..utils.info import INFO, info5
from .utils.repr.distrep import distrep
from .utils.repr.shape import shape5
from .types import MEMPIN, MEMBAT


class VectorSequence(Meta):
    R"""
    Vector sequence meta.
    """
    def __init__(
        self,
        feats_train: onpt.NDArray[onp.generic],
        labels_train: onpt.NDArray[onp.generic],
        feats_valid: onpt.NDArray[onp.generic],
        labels_valid: onpt.NDArray[onp.generic],
        feats_test: onpt.NDArray[onp.generic],
        labels_test: onpt.NDArray[onp.generic],
        /,
    ) -> None:
        R"""
        Initialize the class.
        """
        # Safety check.
        if (
            labels_train.ndim > 0 or labels_valid.ndim > 0
            or labels_test.ndim > 0
        ):
            # UNEXPECT:
            # Label sequence is not supported yet.
            raise NotImplementedError(
                "Label vector sequence is not supported.",
            )

        #
        self.num_seqs_train = len(feats_train)
        self.num_seqs_valid = len(feats_valid)
        self.num_seqs_test = len(feats_test)

        #
        self.feats: onpt.NDArray[onp.generic]

        #
        feats_train = onp.transpose(feats_train, (2, 0, 1))
        feats_valid = onp.transpose(feats_valid, (2, 0, 1))
        feats_test = onp.transpose(feats_test, (2, 0, 1))
        self.feats = (
            onp.concatenate((feats_train, feats_valid, feats_test), axis=1)
        )

        # Collect data shapes.
        self.shapize()

    def shapize(self, /) -> None:
        R"""
        Get data shapes.
        """
        #
        (self.num_times, self.num_seqs, self.feat_size) = self.feats.shape

    def collect(self, /) -> None:
        R"""
        Collect essential statistics.
        """
        # UNEXPECT:
        # No information to be collected.
        raise NotImplementedError("There is no information to be collected.")

    def repr(self, /) -> INFO:
        R"""
        Get essential statistics for representation.
        """
        #
        repdict = Meta.repr(self)
        repdict["Basic"]["#Seqs"] = str(self.num_seqs)
        repdict["Basic"]["#Times"] = str(self.num_times)
        repdict["(Feat)ure"] = (
            {
                "Seq Feat Size": (
                    shape5(
                        str(self.feat_size), self.feat_size, "float64", False,
                    )
                ),
            }
        )
        return repdict

    def __repr__(self, /) -> str:
        R"""
        Get representaiton of the class.
        """
        #
        return info5(self.repr())

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
        begin_train = 0
        end_train = begin_train + self.num_seqs_train
        begin_valid = end_train
        end_valid = begin_valid + self.num_seqs_valid
        begin_test = end_valid
        end_test = begin_test + self.num_seqs_test
        return (
            onp.arange(begin_train, end_train),
            onp.arange(begin_valid, end_valid),
            onp.arange(begin_test, end_test),
        )

    def normalizeby(
        self,
        indices: onpt.NDArray[onp.generic], spindle: str,
        /,
    ) -> List[List[Tuple[float, float]]]:
        R"""
        Normalize using statistics only from given indices.
        """
        #
        return [[(0.0, 1.0)]]

    def inputon(self, on: List[Union[str, List[int]]]) -> None:
        R"""
        Set input feature and label data columns.
        """
        # UNEXPECT:
        # Vector sequence must use full data, thus there is no need to set
        # useful input data columns.
        raise NotImplementedError(
            "There is no need to set input data columns.",
        )

    def targeton(self, on: List[Union[str, List[int]]]) -> None:
        R"""
        Set tatget feature and label data columns.
        """
        # UNEXPECT:
        # Vector sequence must use full data, thus there is no need to set
        # useful target data columns.
        raise NotImplementedError(
            "There is no need to set target data columns.",
        )

    def distrep(self, /, *, n: int) -> str:
        R"""
        Get data distribution representation of the class.
        """
        #
        #
        return (
            info5(
                {
                    "Data (Dist)ribute": {
                        "Feat": distrep(self.feats, axis=(0, 1, 2), n=n)[0],
                    },
                },
            )
        )

    def pin(self, batch_size: int, /) -> MEMPIN:
        R"""
        Always-shared data to be pinned into device memory.
        Shared data should not differentiate input and target.
        """
        # Nothing should be pinned in memory.
        return []

    def __len__(self, /) -> int:
        R"""
        Length of the class.
        """
        #
        return self.num_seqs

    def __getitem__(self, idx: int, /) -> MEMBAT:
        R"""
        Get an indexable item of the class.
        """
        #
        return ([self.feats[:, idx]], [])
