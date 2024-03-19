"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch_geometric.nn import SchNet, radius_graph
from torch_scatter import scatter
from torch_geometric.nn.models.schnet import *

from mdsim.common.registry import registry
from mdsim.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from mdsim.models.pme_utils import *


@registry.register_model("pme_schnet_v4-1")
class PmeSchNetWrap(SchNet):
    r"""Wrapper around the continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_. Each layer uses interaction
    block of the form:

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`128`)
        num_filters (int, optional): Number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): Number of interaction blocks
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
    """

    def __init__(
        self,
        num_atoms,  # not used
        bond_feat_dim,  # not used
        num_targets,
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        mesh_cutoff=10.0,
        mesh_conv_layers=3,
        mesh_partition=20,
        mesh_conv_modes=10,
        mesh_channel=16,
        using_ff=True,
        readout="add",
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.mesh_partition = mesh_partition
        self.mesh_cutoff = mesh_cutoff
        self.mesh_channel = mesh_channel

        super(PmeSchNetWrap, self).__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )

        # fm stands for "for mesh" (sorry for lazy naming)
        self.embedding_fm = Embedding(100, mesh_channel)

        # self.interactions_fm = ModuleList()
        # for _ in range(3):
        #     block = InteractionBlock(
        #         hidden_channels, num_gaussians, num_filters, cutoff
        #     )
        #     self.interactions_fm.append(block)

        self.mesh_interaction = InteractionBlock(
            mesh_channel, num_gaussians, num_filters, cutoff
        )

        self.mesh_interaction2 = InteractionBlock(
            mesh_channel, num_gaussians, num_filters, cutoff
        )

        self.mesh_embedding = nn.Linear(16, mesh_channel)

        self.lin1_fm = Linear(mesh_channel, mesh_channel // 2)
        self.lin2_fm = Linear(mesh_channel // 2, 1)

        self.pmeconv = PMEConv(
            modes1=mesh_conv_modes,
            modes2=mesh_conv_modes,
            modes3=mesh_conv_modes,
            width=mesh_channel,
            num_fourier_time=mesh_channel,
            padding=0,
            num_layers=mesh_conv_layers,
            using_ff=using_ff,
        )

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch

        if self.otf_graph:
            edge_index, cell_offsets, _, neighbors = radius_graph_pbc(
                data, self.cutoff, 500
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long

            out = get_pbc_distances(
                data.pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.natoms,
            )

            edge_index = out["edge_index"]
            edge_weight = out["distances"]
            edge_attr = self.distance_expansion(edge_weight)

            mesh_charge = self.embedding_fm(z)

            mesh = init_particle_mesh(data.cell, self.mesh_partition, self.use_pbc)
            mesh_feat = init_feat(
                data.pos,
                data.atomic_numbers,
                data.batch,
                mesh,
                self.use_pbc,
                n=self.mesh_channel,
                feat_type="zeros",
            )
            mesh_h3 = torch.prod(mesh["mesh_length"], dim=1).unsqueeze(1)

            (
                atom_with_mesh,
                atom_with_super_mesh,
                atom_with_mesh_batch,
                atom_with_mesh_dst,
                atom_with_mesh_src,
                edge_vec,
            ) = process_atom_to_mesh(
                data.pos, data.batch, mesh, self.mesh_cutoff, self.use_pbc
            )

            h_with_mesh = torch.cat([mesh_charge, mesh_feat], dim=0)
            atom_with_mesh_dist = torch.norm(edge_vec, dim=1)
            atom_with_mesh_edge_attr = self.distance_expansion(atom_with_mesh_dist)
            h_with_mesh = self.mesh_interaction(
                h_with_mesh,
                torch.vstack([atom_with_mesh_src, atom_with_mesh_dst]),
                atom_with_mesh_dist,
                atom_with_mesh_edge_attr,
            )
            h_with_mesh *= 1 / mesh_h3[atom_with_mesh_batch]
            mesh_feat = h_with_mesh[mesh_charge.shape[0] :]
            nd = mesh["meshgrid"].shape[1]
            mesh_feat = mesh_feat.reshape(mesh["meshgrid"].shape[0], nd, nd, nd, -1)
            mesh_grid = get_grid(mesh_feat.shape, mesh_feat.device)
            mesh_feat = self.pmeconv(mesh_feat, mesh_grid)
            mesh_feat = mesh_feat.reshape(-1, mesh_feat.shape[-1])

            h_with_mesh = torch.cat([mesh_charge, mesh_feat], dim=0)
            h_with_mesh = self.mesh_interaction2(
                h_with_mesh,
                torch.vstack([atom_with_mesh_dst, atom_with_mesh_src]),
                atom_with_mesh_dist,
                atom_with_mesh_edge_attr,
            )

            mesh_charge = (
                h_with_mesh[: mesh_charge.shape[0]] * mesh_charge * mesh_h3[batch] / (self.mesh_partition-1)**3
            )
            mesh_charge = self.lin1_fm(mesh_charge)
            mesh_charge = self.act(mesh_charge)
            mesh_charge = self.lin2_fm(mesh_charge)

            h = self.embedding(z)

            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)

            h = h + mesh_charge

            batch = torch.zeros_like(z) if batch is None else batch
            energy = scatter(h, batch, dim=0, reduce=self.readout)
        else:
            # Not implemented yet
            raise NotImplementedError

            energy = super(PmeSchNetWrap, self).forward(z, pos, batch)
        return energy

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return energy, forces
        else:
            return energy

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
