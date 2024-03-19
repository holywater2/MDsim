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


@registry.register_model("pme_charge_gnn")
class PmeGNN(torch.nn.Module):
    def __init__(
        self,
        num_targets,
        use_pbc=True,
        regress_forces=True,
        otf_graph=False,
        cutoff=5.0,
        using_ff=True,
        mesh_cutoff=5.0,
        mesh_conv_layers=3,
        mesh_partition=20,
        mesh_conv_modes=10,
        mesh_channel=32,
        num_spherical=3,
        num_radial=16,
        num_radial_layer=2,
        radial_embed_size=64,
        radial_hidden_size=128,
        num_sh_gcn_layers=3,
        mesh_hidden_channel=16,
        mesh_feat_type="zeros",
        readout="add",
        gauss_start=0.0,
        gauss_end=5.0,
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.readout = readout

        self.mesh_channel = mesh_channel
        self.mesh_partition = mesh_partition
        self.mesh_cutoff = mesh_cutoff

        self.num_radial = num_radial
        self.num_spherical = num_spherical

        self.num_radial_layer = num_radial_layer
        self.radial_embed_size = radial_embed_size
        self.radial_hidden_size = radial_hidden_size
        self.num_sh_gcn_layers = num_sh_gcn_layers
        self.mesh_hidden_channel = mesh_hidden_channel
        self.mesh_feat_type = mesh_feat_type

        super(PmeGNN, self).__init__()
        # fm stands for "for mesh" (sorry for lazy naming)
        self.embedding_sh = nn.Embedding(100, self.num_radial)
        self.irreps_sh = o3.Irreps.spherical_harmonics(self.num_spherical, p=1)
        self.irreps_feat = (self.irreps_sh * self.num_radial).sort().irreps.simplify()

        self.SphericalGCNs = nn.ModuleList(
            [
                SphericalGCN(
                    (f"{num_radial}x0e" if i == 0 else self.irreps_feat),
                    self.irreps_feat,
                    self.irreps_sh,
                    radial_embed_size,
                    num_radial_layer,
                    radial_hidden_size,
                    is_fc=True,
                )
                for i in range(num_sh_gcn_layers)
            ]
        )
        self.orbital = GaussianOrbital(
            gauss_start, gauss_end, num_radial, num_spherical
        )

        self.act_sh = NormActivation(self.irreps_feat)

        self.atom_to_mesh_gcn = SphericalGCN(
            self.irreps_feat,
            f"{self.mesh_channel}x0e",
            self.irreps_sh,
            radial_embed_size,
            num_radial_layer,
            radial_hidden_size,
            is_fc=True,
            use_sc=False,
        )
        self.act_mesh = NormActivation(f"{self.mesh_channel}x0e")

        self.mesh_to_atom_gcn = SphericalGCN(
            f"{self.mesh_hidden_channel}x0e",
            f"0e",
            self.irreps_sh,
            radial_embed_size,
            num_radial_layer,
            radial_hidden_size,
            is_fc=True,
            use_sc=False,
        )

        self.pmeconv = PMEConv(
            modes1=mesh_conv_modes,
            modes2=mesh_conv_modes,
            modes3=mesh_conv_modes,
            width=mesh_hidden_channel,
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
        num_batch = data.cell.size(0)

        if self.otf_graph:
            edge_index, cell_offsets, _, neighbors = radius_graph_pbc(
                data, self.cutoff, 500
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc:
            assert z.dim() == 1 and z.dtype == torch.long

            z = data.atomic_numbers.long()
            edge_index = radius_graph(data.pos, self.cutoff, batch, loop=False)
            src, dst = edge_index
            edge_vec = data.pos[src] - data.pos[dst]
            edge_len = edge_vec.norm(dim=-1) + 1e-8
            edge_feat = o3.spherical_harmonics(
                list(range(self.num_spherical + 1)),
                edge_vec / edge_len[..., None],
                normalize=False,
                normalization="integral",
            )
            edge_embed = soft_one_hot_linspace(
                edge_len,
                start=0.0,
                end=self.cutoff,
                number=self.radial_embed_size,
                basis="gaussian",
                cutoff=False,
            ).mul(self.radial_embed_size**0.5)

            atom_feat = self.embedding_sh(z)
            for i, gcn in enumerate(self.SphericalGCNs):
                atom_feat = gcn(
                    edge_index,
                    atom_feat,
                    edge_feat,
                    edge_embed,
                    dim_size=data.pos.size(0),
                )
                if i != self.num_sh_gcn_layers - 1:
                    atom_feat = self.act_sh(atom_feat)

            mesh = init_particle_mesh(data.cell, self.mesh_partition, self.use_pbc)
            mesh_feat = init_feat(
                data.pos,
                data.atomic_numbers,
                data.batch,
                mesh,
                self.use_pbc,
                n=15,
                feat_type=self.mesh_feat_type,
            )

            mesh_dst, atom_src, edge_vec = radius_and_edge_atom_to_mesh(
                data.pos, data.batch, mesh, self.mesh_cutoff, self.use_pbc
            )

            density_edge = mesh["meshgrid"].reshape(num_batch, -1, 3)[
                batch
            ] - data.pos.unsqueeze(-2)

            density = (self.orbital(density_edge) * atom_feat.unsqueeze(1)).sum(dim=-1)
            density = scatter(density, batch, dim=0, reduce="sum")
            density = density.transpose(0, 1)
            mesh_feat = torch.cat([mesh_feat, density], dim=-1)

            # atm stands for atom to mesh
            atm_len = torch.norm(edge_vec, dim=-1) + 1e-8
            atm_edge_feat = o3.spherical_harmonics(
                list(range(self.num_spherical + 1)),
                edge_vec / (atm_len[..., None] + 1e-8),
                normalize=False,
                normalization="integral",
            )
            atm_edge_embed = soft_one_hot_linspace(
                atm_len,
                start=0.0,
                end=self.cutoff,
                number=self.radial_embed_size,
                basis="gaussian",
                cutoff=False,
            ).mul(self.radial_embed_size**0.5)

            mesh_feat = self.atom_to_mesh_gcn.forward(
                (atom_src, mesh_dst),
                atom_feat,
                atm_edge_feat,
                atm_edge_embed,
                dim_size=mesh["meshgrid_flat"].size(0),
            )

            mesh_feat = self.act_mesh(mesh_feat)

            mesh_feat = mesh_feat.reshape(
                mesh["meshgrid"].shape[0],
                self.mesh_partition,
                self.mesh_partition,
                self.mesh_partition,
                -1,
            )

            mesh_grid = get_grid(mesh_feat.shape, mesh_feat.device)
            mesh_feat = self.pmeconv(mesh_feat, mesh_grid)
            mesh_feat = mesh_feat.reshape(-1, mesh_feat.shape[-1])

            atom_feat = self.mesh_to_atom_gcn.forward(
                (mesh_dst, atom_src),
                mesh_feat,
                atm_edge_feat,
                atm_edge_embed,
                dim_size=data.pos.size(0),
            )

            batch = torch.zeros_like(z) if batch is None else batch
            energy = scatter(atom_feat, batch, dim=0, reduce=self.readout)
        else:
            # Not implemented yet
            raise NotImplementedError

            energy = super(PmeGNN, self).forward(z, pos, batch)
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
