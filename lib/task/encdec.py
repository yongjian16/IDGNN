R"""
"""
#
import torch
from typing import Tuple, List
from .task import Task
from ..model.activate import activatize
from ..model.initialize import glorot
from ..model.snn import sequentialize
from ..model.mlp import MLP
from .regression import mse_loss, metrics


class SelfEncoderDecoder(Task):
    R"""
    Self-supervised encoder-decoder.
    """
    def __init__(
        self,
        feat_size: int, embed_inside_size: int,
        /,
        *,
        reduce: str, activate: str,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Task.__init__(self)

        #
        self.feat_size = feat_size

        #
        self.encoder = sequentialize(reduce, feat_size, embed_inside_size)
        self.decoder = sequentialize(reduce, 1, embed_inside_size)
        self.mlp = (
            MLP(
                embed_inside_size, feat_size, embed_inside_size,
                activate=activate,
            )
        )
        self.activate = activatize(activate)

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        resetted = resetted + glorot(self.encoder, rng)
        resetted = resetted + glorot(self.decoder, rng)
        resetted = resetted + self.mlp.reset(rng)
        return resetted

    def reshape(
        self,
        /,
        *ARGS,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        R"""
        Reshape tensors in given arguments for model forwarding.
        """
        #
        (_, (feats,)) = ARGS
        return ([feats], [feats])

    def forward(self, input: torch.Tensor, /) -> List[torch.Tensor]:
        R"""
        Forward.
        """
        #
        (num_times, num_seqs, num_feats) = input.shape
        (output, _) = self.encoder.forward(input)
        virtual = (
            torch.zeros(
                num_times, num_seqs, 1,
                dtype=input.dtype, device=input.device,
            )
        )
        (output, _) = self.decoder.forward(virtual, output[[-1]])
        (_, _, num_hiddens) = output.shape
        output = torch.reshape(output, (num_times * num_seqs, num_hiddens))
        output = self.mlp.forward(output)
        output = torch.reshape(output, (num_times, num_seqs, num_feats))
        return [output]

    def loss(self, /, *ARGS) -> torch.Tensor:
        R"""
        Loss funtion.
        """
        #
        output_feats: torch.Tensor
        target_feats: torch.Tensor

        # Output and target only have feature data.
        # Target node label data are not useful in this task.
        (output_feats, target_feats) = ARGS
        (num_times_output, num_seqs_output, _) = output_feats.shape
        (num_times_target, num_seqs_target, _) = target_feats.shape

        # Format output and target data.
        output_feats = (
            torch.reshape(
                output_feats,
                (num_times_output * num_seqs_output, self.feat_size),
            )
        )
        target_feats = (
            torch.reshape(
                target_feats,
                (num_times_target * num_seqs_target, self.feat_size),
            )
        )
        return mse_loss(output_feats, target_feats)

    def metrics(self, /, *ARGS) -> List[Tuple[int, float]]:
        R"""
        Evaluation metrics.
        """
        #
        output_feats: torch.Tensor
        target_feats: torch.Tensor

        # Output and target only have feature data.
        # Target node label data are not useful in this task.
        (output_feats, target_feats) = ARGS
        (num_times_output, num_seqs_output, _) = output_feats.shape
        (num_times_target, num_seqs_target, _) = target_feats.shape

        # Format output and target data.
        output_feats = (
            torch.reshape(
                output_feats,
                (num_times_output * num_seqs_output, self.feat_size),
            )
        )
        target_feats = (
            torch.reshape(
                target_feats,
                (num_times_target * num_seqs_target, self.feat_size),
            )
        )
        return metrics(output_feats, target_feats)