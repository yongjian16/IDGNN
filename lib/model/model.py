R"""
"""
#
import abc
import torch
import numpy as onp
from typing import Tuple, List, Dict
from ..utils.info import noescape, infotab5


class Model(abc.ABC, torch.nn.Module):
    R"""
    Model.
    """
    #
    COSTS: Dict[str, List[float]]

    #
    COSTS = {"graph": [], "non-graph": [], "edges": []}

    # Signal to use simplest model for greatest efficency on synthetic tasks.
    SIMPLEST = False

    def __annotation__(self, /) -> None:
        R"""
        Annotate for class instance attributes.
        """
        #
        self.feat_target_size: int

    @abc.abstractmethod
    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        ...

    def initialize(self, seed: int, /) -> None:
        R"""
        Explicitly initialize the model.
        """
        #
        rng = torch.Generator("cpu")
        rng.manual_seed(seed)
        resetted = self.reset(rng)
        if resetted != sum(param.numel() for param in self.parameters()):
            # UNEXPECT:
            # All defined parameters should match exactly with initialization.
            raise NotImplementedError(
                "Defined parameters do not exactly match with initialized "
                "parameters.",
            )
        self.num_resetted_params = resetted

    def __repr__(self) -> str:
        R"""
        Get representation of the class.
        """
        # Model parameter key info has a special visible representation.
        names = []
        shapes = []
        for name, param in self.named_parameters():
            #
            names.append(name.split("."))
            shapes.append(
                "\x1b[90mx\x1b[0m".join(str(dim) for dim in param.shape),
            )
        depth = 0 if len(names) == 0 else max(len(levels) for levels in names)
        padded = [levels + [""] * (depth - len(levels)) for levels in names]

        #
        keys = (
            "\x1b[90m-\x1b[92m→\x1b[0m".join(levels).replace(
                "\x1b[92m→\x1b[0m\x1b[90m", "\x1b[90m→",
            )
            for levels in (
                [
                    [
                        "{:<{:d}s}".format(name, maxlen).replace(
                            " ", "\x1b[90m-\x1b[0m",
                        )
                        for (name, maxlen) in (
                            zip(
                                levels,
                                (
                                    max(len(name) for name in level)
                                    for level in zip(*padded)
                                ),
                            )
                        )
                    ]
                    for levels in padded
                ]
            )
        )

        # We may also care about the product besides shape.
        maxlen = (
            0
            if len(shapes) == 0 else
            max(len(noescape(shape)) for shape in shapes)
        )
        shapes = (
            [
                "{:s}{:s} ({:d})".format(
                    " " * (maxlen - len(noescape(shape))), shape,
                    int(
                        onp.prod(
                            [
                                int(dim)
                                for dim in shape.split("\x1b[90mx\x1b[0m")
                            ],
                        ),
                    ),
                )
                for shape in shapes
            ]
        )

        # Generate final representation.
        return "\n".join(infotab5(
            "(Param)eter",
            [
                key + "\x1b[90m→\x1b[94m:\x1b[0m " + shape
                for (key, shape) in zip(keys, shapes)
            ],
        ))

    def moveon(self, notembedon: List[int]) -> None:
        R"""
        Set axes for moving window model.
        """
        # EXPECT:
        # By default, the model is not moving window unless it is overloaded.
        raise RuntimeError(
            "Default model is not a moving window model, and you need to "
            "explicitly overload to use moving window.",
        )

    def pretrain(self, partname: str, path: str, /) -> None:
        R"""
        Use pretrained model.
        """
        # EXPECT:
        # By default, there is no pretraining definition.
        raise RuntimeError(
            "No pretraining of \"{:s}\" is defined."
            .format(self.__class__.__name__),
        )