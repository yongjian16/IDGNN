R"""
"""
#
import torch


def activatize(name: str, /) -> torch.nn.Module:
    R"""
    Get activation module.
    """
    #
    if name == "softplus":
        #
        return torch.nn.Softplus()
    elif name == "sigmoid":
        #
        return torch.nn.Sigmoid()
    elif name == "tanh":
        #
        return torch.nn.Tanh()
    elif name == "identity":
        #
        return torch.nn.Identity()
    elif name == "logsoftmax":
        return torch.nn.LogSoftmax(dim=-1)
    if name == "softmax":
        #
        return torch.nn.Softmax(dim=-1)
    else:
        # EXPECT:
        # It is possible to require unsupporting sequential model.
        raise RuntimeError(
            "Activation module identifier \"{:s}\" is not supported."
            .format(name),
        )