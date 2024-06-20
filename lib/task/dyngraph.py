R"""
"""
#
import torch
from typing import Tuple, List
from .task import Task


class DynamicGraph(Task):
    R"""
    Task on generic dynamic graph.
    """
    #
    DYNEDGE = False

    def dynedge(self, /) -> None:
        R"""
        Support dynamic edge.
        """
        #
        self.DYNEDGE = True

    def reshape(
        self,
        /,
        *ARGS,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        R"""
        Reshape tensors in given arguments for model forwarding.
        """
        # Parse unformatted batch data.
        if self.DYNEDGE:
            #
            (
                _,
                (
                    edge_tuples, edge_feats, edge_labels, edge_ranges,
                    edge_times, node_feats_input, node_labels_input,
                    node_times_input, node_feats_target, node_labels_target,
                ),
                node_masks,
            ) = ARGS
        else:
            #
            (
                (
                    edge_tuples, edge_feats, edge_labels, edge_ranges,
                    edge_times,
                ),
                (
                    node_feats_input, node_labels_input, node_times_input,
                    node_feats_target, node_labels_target,
                ),
                node_masks,
            ) = ARGS
        if node_feats_input.ndim == 3:
            #
            (_, _, num_times_input) = node_feats_input.shape
        else:
            #
            (_, _, num_times_input) = node_labels_input.shape
        if node_feats_target.ndim == 3: # tensor(0., device='cuda:0')
            #
            (_, _, num_times_target) = node_feats_target.shape
        else:
            #
            if (
                node_labels_target.ndim == 2 # tensor([], device='cuda:0', size=(2934, 1, 0), dtype=torch.int64)
                or (
                    node_labels_target.ndim == 3
                    and node_labels_target.shape[1] == 1
                )
            ):
                #
                node_labels_target = (
                    torch.reshape(
                        node_labels_target, (len(node_labels_target), 1, 1),
                    )
                )
                
            else:
                # EXPECT:
                # Only expect static node target labels.
                raise RuntimeError("Only expect static node target labels.")
            (_, _, num_times_target) = node_labels_target.shape

        def reshapenode(tensor: torch.Tensor, n: int, /) -> torch.Tensor:
            R"""
            Reshape node into proper shape.
            """
            if tensor.ndim == 2:
                # Static node data need broadcasting.
                tensor = torch.reshape(tensor, (*tensor.shape, 1))
                return tensor.expand(1, 1, n)
            else:
                # Other cases are assumed to be vaid dynamic node data.
                # For example, already 3D tensor or meaningless scalar place
                # holder.
                return tensor

        # Expand static nodes into dynamic nodes if necessary.
        node_feats_input = reshapenode(node_feats_input, num_times_input)
        node_labels_input = reshapenode(node_labels_input, num_times_input)
        node_feats_target = reshapenode(node_feats_target, num_times_target)
        node_labels_target = reshapenode(node_labels_target, num_times_target)
        return (
            [
                edge_tuples, edge_feats, edge_labels, edge_ranges, edge_times,
                node_feats_input, node_labels_input, node_times_input,
                node_masks,
            ],
            [node_feats_target, node_labels_target, node_masks],
        )