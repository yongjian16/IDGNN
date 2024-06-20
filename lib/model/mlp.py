R"""
"""
#
import torch
from .model import Model
from .activate import activatize
from .initialize import glorot


class MLP(Model):
    R"""
    Multi-layer perceptron.
    """
    def __init__(
        self,
        feat_input_size: int, feat_target_size: int, embed_inside_size: int,
        /,
        *,
        activate: str,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        #
        self.lin1 = torch.nn.Linear(feat_input_size, embed_inside_size)
        self.lin2 = torch.nn.Linear(embed_inside_size, feat_target_size)
        self.activate = activatize(activate)

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        resetted = resetted + glorot(self.lin1, rng)
        resetted = resetted + glorot(self.lin2, rng)
        return resetted

    def forward(self, input: torch.Tensor, /) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        embed = self.lin1.forward(input)
        embed = self.lin2.forward(self.activate(embed))
        return embed