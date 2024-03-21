"""
adapted from:
https://github.com/Open-Catalyst-Project/ocp
"""

import torch
from torch_geometric.nn import radius_graph
from torch_scatter import scatter

from mdsim.models.utils.dimenet_plus_plus_layers import (
    DimeNetPlusPlus,
    InteractionPPBlock,
    OutputPPBlock,
    swish,
)
from mdsim.common.registry import registry
from mdsim.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from torch_geometric.nn.models.schnet import *

from mdsim.models.pme_utils import *
from neuralop.models import FNO


@registry.register_model("dimenetplusplus_pme_v3")
class DimeNetPlusPlusWrap(DimeNetPlusPlus):
    def __init__(
        self,
        num_atoms,
        bond_feat_dim,  # not used
        num_targets,
        use_pbc=True,
        regress_forces=True,
        hidden_channels=128,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        otf_graph=False,
        cutoff=10.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        mesh_cutoff=10.0,
        # mesh_conv_layers=3,
        mesh_partition=20,
        # mesh_conv_modes=10,
        mesh_channel=32,
        # using_ff=False,
        num_gaussians=50,
        num_filters=128,
        sch_cutoff=6.0,
        *args,
        **kwargs,
    ):
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph

        self.mesh_channel = mesh_channel
        self.mesh_partition = mesh_partition
        self.mesh_cutoff = mesh_cutoff

        super(DimeNetPlusPlusWrap, self).__init__(
            hidden_channels=hidden_channels,
            out_channels=num_targets,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
        )

        self.embedding_fm = Embedding(100, mesh_channel)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.act = ShiftedSoftplus()

        self.mesh_interaction = InteractionBlock(
            mesh_channel, num_gaussians, num_filters, sch_cutoff
        )

        # self.mesh_interaction2 = InteractionBlock(
        #     mesh_channel, num_gaussians, num_filters, sch_cutoff
        # )

        self.mesh_embedding = nn.Linear(16, mesh_channel)

        self.lin1_fm = Linear(mesh_channel, mesh_channel // 2)
        self.lin2_fm = Linear(mesh_channel // 2, 1)

        # self.pmeconv = PMEConv(
        #     modes1=mesh_conv_modes,
        #     modes2=mesh_conv_modes,
        #     modes3=mesh_conv_modes,
        #     width=mesh_channel,
        #     num_fourier_time=mesh_channel,
        #     padding=0,
        #     num_layers=mesh_conv_layers,
        #     using_ff=using_ff,
        # )

        self.pmeconv = FNO(
            n_modes=(mesh_partition, mesh_partition, mesh_partition),
            in_channels=mesh_channel + 6,
            out_channels=mesh_channel,
            hidden_channels=32,
            projection_channels=64,
            factorization="tucker",
            rank=0.42,
        )

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        pos = data.pos
        batch = data.batch

        if self.otf_graph:
            edge_index, cell_offsets, _, neighbors = radius_graph_pbc(
                data, self.cutoff, 50
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors

        if self.use_pbc:
            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.natoms,
                return_offsets=True,
            )

            edge_index = out["edge_index"]
            dist = out["distances"]
            offsets = out["offsets"]

            j, i = edge_index
        else:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
            j, i = edge_index
            dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        _, _, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index,
            data.cell_offsets,
            num_nodes=data.atomic_numbers.size(0),
        )

        # Calculate angles.
        pos_i = pos[idx_i].detach()
        pos_j = pos[idx_j].detach()
        if self.use_pbc:
            pos_ji, pos_kj = (
                pos[idx_j].detach() - pos_i + offsets[idx_ji],
                pos[idx_k].detach() - pos_j + offsets[idx_kj],
            )
        else:
            pos_ji, pos_kj = (
                pos[idx_j].detach() - pos_i,
                pos[idx_k].detach() - pos_j,
            )

        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(data.atomic_numbers.long(), rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:]
        ):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i, num_nodes=pos.size(0))

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
        # mesh_h3 = torch.prod(mesh["mesh_length"], dim=1).unsqueeze(1)

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

        z = data.atomic_numbers.long()
        mesh_atom_feat = self.embedding_fm(z)

        h_with_mesh = torch.cat([mesh_atom_feat, mesh_feat], dim=0)
        atom_with_mesh_dist = torch.norm(edge_vec, dim=1)
        atom_with_mesh_edge_attr = self.distance_expansion(atom_with_mesh_dist)

        h_with_mesh = self.mesh_interaction(
            h_with_mesh,
            torch.vstack([atom_with_mesh_src, atom_with_mesh_dst]),
            atom_with_mesh_dist,
            atom_with_mesh_edge_attr,
        )
        mesh_charge = h_with_mesh[mesh_atom_feat.shape[0] :]

        # h_with_mesh *= 1 / mesh_h3[atom_with_mesh_batch]
        mesh_feat = h_with_mesh[mesh_atom_feat.shape[0] :]
        nd = mesh["meshgrid"].shape[1]
        mesh_feat = mesh_feat.reshape(mesh["meshgrid"].shape[0], nd, nd, nd, -1)
        mesh_grid = get_grid(mesh_feat.shape, mesh_feat.device)
        mesh_feat = torch.cat([mesh_feat, mesh["meshgrid"], mesh_grid], dim=-1)
        mesh_feat = mesh_feat.permute(0, 4, 1, 2, 3)
        # mesh_feat = self.pmeconv(mesh_feat, mesh_grid)
        mesh_feat = self.pmeconv(mesh_feat)
        mesh_feat = mesh_feat.permute(0, 2, 3, 4, 1)
        mesh_feat = mesh_feat.reshape(-1, mesh_feat.shape[-1])

        h_with_mesh = torch.cat([mesh_atom_feat, mesh_feat], dim=0)

        # mesh_feat = mesh_feat * mesh_charge * mesh_h3[mesh["meshgrid_batch"]]
        mesh_feat = mesh_feat * mesh_charge

        mesh_charge = self.lin1_fm(mesh_charge)
        mesh_charge = self.act(mesh_charge)
        mesh_charge = self.lin2_fm(mesh_charge)

        mesh_energy = scatter(mesh_charge, mesh["meshgrid_batch"], dim=0)

        # h_with_mesh = torch.cat([mesh_charge, mesh_feat], dim=0)
        # atom_with_mesh_dist = torch.norm(edge_vec, dim=1)
        # atom_with_mesh_edge_attr = self.distance_expansion(atom_with_mesh_dist)
        # h_with_mesh = self.mesh_interaction(
        #     h_with_mesh,
        #     torch.vstack([atom_with_mesh_src, atom_with_mesh_dst]),
        #     atom_with_mesh_dist,
        #     atom_with_mesh_edge_attr,
        # )
        # h_with_mesh *= 1 / mesh_h3[atom_with_mesh_batch]
        # mesh_feat = h_with_mesh[mesh_charge.shape[0] :]
        # nd = mesh["meshgrid"].shape[1]
        # mesh_feat = mesh_feat.reshape(mesh["meshgrid"].shape[0], nd, nd, nd, -1)
        # mesh_grid = get_grid(mesh_feat.shape, mesh_feat.device)
        # mesh_feat = self.pmeconv(mesh_feat, mesh_grid)
        # mesh_feat = mesh_feat.reshape(-1, mesh_feat.shape[-1])

        # h_with_mesh = torch.cat([mesh_charge, mesh_feat], dim=0)
        # h_with_mesh = self.mesh_interaction2(
        #     h_with_mesh,
        #     torch.vstack([atom_with_mesh_dst, atom_with_mesh_src]),
        #     atom_with_mesh_dist,
        #     atom_with_mesh_edge_attr,
        # )

        # mesh_charge = h_with_mesh[: mesh_charge.shape[0]] * mesh_charge * mesh_h3[batch]
        # mesh_charge = self.lin1_fm(mesh_charge)
        # mesh_charge = self.act(mesh_charge)
        # mesh_charge = self.lin2_fm(mesh_charge)

        energy = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)
        energy += mesh_energy

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
