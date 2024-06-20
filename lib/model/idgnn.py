R"""
"""
#
import abc
import torch
import torch_geometric as thgeo
from typing import List, cast
from .model import Model
from .initialize import glorot

import math
import torch.nn as nn
import numpy as onp
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch.nn import Parameter
from .utils import projection_norm_inf

class IDGNN(Model):
    R"""
    Implicit Dynamic Graph neural network.
    """
    def __init__(
        self,
        feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int, embed_inside_size: int,
        /,
        *,
        convolve: str, skip: bool, activate: str,
        num_nodes, window_size, kappa=0.99, 
        phi=None, multi_z=True, multi_x=False, 
        regression=False,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        self.k = kappa
        self.embed_inside_size = embed_inside_size
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.regression = regression
        self.feat_target_size = feat_target_size

        if multi_z:
            self.gcn_z = nn.ModuleList([GCNConv(embed_inside_size, embed_inside_size, normalize=True, bias=False) 
                                        for times in range(window_size)])
        else:
            self.gcn_z = nn.ModuleList([GCNConv(embed_inside_size, embed_inside_size, normalize=True, bias=False)])

        # if multi_x:
        #     self.gcn_x = nn.ModuleList([GCNConv(feat_input_size_node, embed_inside_size, normalize=True) 
        #                                 for times in range(window_size)])
        # else:
        #     self.gcn_x = nn.ModuleList([GCNConv(feat_input_size_node, embed_inside_size, normalize=True)])
        # self.Z_0 = Parameter(torch.zeros(embed_inside_size, num_nodes), requires_grad=False)

        if multi_x:
            self.lin_x = nn.ModuleList([nn.Linear(feat_input_size_node, embed_inside_size, bias=True) 
                                        for times in range(window_size)])
        else:
            self.lin_x = nn.ModuleList([nn.Linear(feat_input_size_node, embed_inside_size, bias=True)])

        self.Z_0 = Parameter(torch.zeros(num_nodes, embed_inside_size), requires_grad=False)
        
        if phi is None:
            self.phi = F.relu
        else:
            self.phi = phi

    def init(self):
        numels = 0
        stdv = 1 / math.sqrt(self.gcn_z[0].lin.weight.shape[0])
        for gcn in self.gcn_z:
            gcn.lin.weight.data.uniform_(-stdv, stdv)
            numels += gcn.lin.weight.numel()
        if hasattr(self, 'gcn_x'):
            stdv = 1 / math.sqrt(self.gcn_x[0].lin.weight.shape[0])
            for gcn in self.gcn_x:
                gcn.lin.weight.data.uniform_(-stdv, stdv)
                numels += gcn.lin.weight.numel()
        elif hasattr(self, 'lin_x'):
            stdv = 1 / math.sqrt(self.lin_x[0].weight.shape[0])
            for lin in self.lin_x:
                lin.weight.data.uniform_(0, stdv)
                numels += lin.weight.numel()
        return numels
    
    def glorot(self, module, rng):
        if isinstance(module, torch.nn.Linear):
            module = cast(torch.nn.Linear, module)
            weight = module.weight
            bias = module.bias
            # (fan_out, fan_in) = weight.shape
            # a = onp.sqrt(6 / (fan_in + fan_out))
            a = 1 / onp.sqrt(weight.shape[0])
            #
            weight.data.uniform_(-a, a, generator=rng)
            resetted = weight.numel()

            # Pytorch always annotate bias as an existing object while it may be None.
            # Use attribute-checking to pass static typing.
            if hasattr(bias, "numel"):
                # May not have bias.
                bias.data.uniform_(-a, a, generator=rng)
                resetted = resetted + bias.numel()
            else:
                #
                resetted = resetted + 0
            return resetted
        
        elif isinstance(module, thgeo.nn.GCNConv):

            #
            weight: torch.nn.parameter.Parameter
            bias: torch.nn.parameter.Parameter

            #
            module = cast(thgeo.nn.GCNConv, module)
            weight = getattr(getattr(module, "lin"), "weight")
            bias = getattr(module, "bias")
            a = 1 / onp.sqrt(weight.shape[0])
            weight.data.uniform_(-a, a, generator=rng)
            resetted = weight.numel()
            if hasattr(bias, "numel"):
                # May not have bias.
                bias.data.uniform_(-a, a, generator=rng)
                resetted = resetted + bias.numel()
            else:
                #
                resetted = resetted + 0
            return resetted
        else:
            raise NotImplementedError("Unknown module type.")

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.

        """
        #
        # resetted = self.init()
        resetted = 0
        for gcn in self.gcn_z:
            resetted = resetted + self.glorot(gcn, rng)
        # for gcn in self.gcn_x:
        #     resetted = resetted + glorot(gcn, rng)
        for lin in self.lin_x:
            resetted = resetted + self.glorot(lin, rng)

        self.Z_0.data.uniform_(0, 1)
        resetted += self.Z_0.numel()
        return resetted
    
    def project(self, A_rho=None):
        if A_rho is not None:
            for i, gcn in enumerate(self.gcn_z):
                gcn.lin.weight.data = projection_norm_inf(gcn.lin.weight.data, 
                                                          kappa=self.k / A_rho[i])

    def _forward(self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        edge_ranges: torch.Tensor, edge_times: torch.Tensor,
        node_feats: torch.Tensor, node_times: torch.Tensor,
        node_masks: torch.Tensor,
        /,
        Z:torch.Tensor=None,
    ) -> torch.Tensor:
        (_, num_times) = edge_ranges.shape

        # Forward
        for i in range(num_times):
            edge_snap_slice = slice(edge_ranges[0, i], edge_ranges[1, i])
            X, A, Aw = node_feats[:, :, i], edge_tuples[:, edge_snap_slice], edge_feats[edge_snap_slice, :]

            support1 = self.gcn_z[i](Z, A, Aw) if len(self.gcn_z)>1 else self.gcn_z[0](Z, A, Aw)
            support2 = self.lin_x[i](X) if len(self.lin_x)>1 else self.lin_x[0](X)
            # support2 = self.gcn_x[i](X, A) if len(self.gcn_x)>1 else self.gcn_x[0](X, A)
            Z = self.phi(support1 + support2)

        return Z
    
    def forward(self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        edge_ranges: torch.Tensor, edge_times: torch.Tensor,
        node_feats: torch.Tensor, node_times: torch.Tensor,
        node_masks: torch.Tensor,
        /,
        Z:torch.Tensor=None,
        A_rho:onp.ndarray=None,
    ) -> torch.Tensor:
        
        self.train()
        if A_rho.ndim > 1:
            # if running batch, taking the min value over samples for each timestep
            A_rho = A_rho.min(axis=0)

        self.project(A_rho)

        return self._forward(edge_tuples, edge_feats, edge_ranges, edge_times,
                        node_feats, node_times, node_masks, Z)
    
    @torch.no_grad()
    def predict(self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        edge_ranges: torch.Tensor, edge_times: torch.Tensor,
        node_feats: torch.Tensor, node_times: torch.Tensor,
        node_masks: torch.Tensor,
        /,
        Z:torch.Tensor=None,
        max_iter=500, tol=5e-6,
    ) -> torch.Tensor:

        if Z is None:
            Z = self.Z_0
        self.eval()
        converged = False
        for i in range(max_iter):
            Z_old = Z
            Z = self._forward(edge_tuples, edge_feats, edge_ranges, edge_times, 
                             node_feats, node_times, node_masks, Z)
            err = torch.norm(Z - Z_old, onp.inf)
            if err < tol:
                converged = True
                break
        if not converged:
            print('Warning: IDGNN did not converge after {} iterations'.format(max_iter))
                
        return Z