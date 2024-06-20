R"""
"""
#
import abc
import torch
from typing import Tuple, List, Union
from ..model.model import Model


class Task(Model):
    R"""
    Task.
    """
    @abc.abstractmethod
    def reshape(
        self,
        /,
        *ARGS,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        R"""
        Reshape tensors in given arguments for model forwarding.
        """
        #
        ...

    @abc.abstractmethod
    def loss(self, /, *ARGS) -> torch.Tensor:
        R"""
        Loss funtion.
        """
        #
        ...

    def sidestep(self, /, *ARGS) -> None:
        R"""
        Forward and backward.
        """
        #
        torch.autograd.set_detect_anomaly(True)
        loss = self.loss(*ARGS)
        loss.backward(retain_graph=True)
        return loss

    @abc.abstractmethod
    def metrics(self, /, *ARGS) -> List[Tuple[int, float]]:
        R"""
        Evaluation metrics.
        """
        #
        ...