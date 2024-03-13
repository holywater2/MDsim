"""
adapted from:
https://github.com/Open-Catalyst-Project/ocp
"""

import torch
from torch import nn
from torch_geometric.nn import radius_graph
from math import pi as PI

from torch_geometric.nn.acts import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.dimenet import (
    EmbeddingBlock,
    ResidualLayer,
    SphericalBasisLayer,
)
from torch_scatter import scatter
from torch_sparse import SparseTensor

from mdsim.common.registry import registry
from mdsim.common.utils import (
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

try:
    import sympy as sym
except ImportError:
    sym = None

from mdsim.models.pme_utils import *


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class BesselBasisLayer(torch.nn.Module):
    def __init__(
        self, num_radial: int, cutoff: float = 5.0, envelope_exponent: int = 5
    ):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.empty(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class InteractionPPBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act=swish,
    ):
        super(InteractionPPBlock, self).__init__()
        self.act = act

        # Transformations of Bessel and spherical basis representations.
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(
            num_spherical * num_radial, basis_emb_size, bias=False
        )
        self.lin_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)

        # Dense transformations of input messages.
        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets.
        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection.
        self.layers_before_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)]
        )
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)]
        )

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        # Initial transformations.
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis.
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down-project embeddings and generate interaction triplet embeddings.
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis.
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings.
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputPPBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_layers,
        act=swish,
    ):
        super(OutputPPBlock, self).__init__()
        self.act = act

        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)
        self.lin_up = nn.Linear(hidden_channels, out_emb_channels, bias=True)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(nn.Linear(out_emb_channels, out_emb_channels))
        self.lin = nn.Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def forward(self, x, rbf, i, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class DimeNetPlusPlus(torch.nn.Module):
    r"""DimeNet++ implementation based on https://github.com/klicperajo/dimenet.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Embedding size used for interaction triplets
        basis_emb_size (int): Embedding size used in the basis transformation
        out_emb_channels(int): Embedding size used for atoms in the output block
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (function, optional): The activation funtion.
            (default: :obj:`swish`)
    """

    url = "https://github.com/klicperajo/dimenet/raw/master/pretrained"

    def __init__(
        self,
        hidden_channels,
        out_channels,
        num_blocks,
        int_emb_size,
        basis_emb_size,
        out_emb_channels,
        num_spherical,
        num_radial,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=swish,
    ):
        super(DimeNetPlusPlus, self).__init__()

        self.cutoff = cutoff

        if sym is None:
            raise ImportError("Package `sympy` could not be found.")

        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(
            num_spherical, num_radial, cutoff, envelope_exponent
        )

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = torch.nn.ModuleList(
            [
                OutputPPBlock(
                    num_radial,
                    hidden_channels,
                    out_emb_channels,
                    out_channels,
                    num_output_layers,
                    act,
                )
                for _ in range(num_blocks + 1)
            ]
        )

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionPPBlock(
                    hidden_channels,
                    int_emb_size,
                    basis_emb_size,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def triplets(self, edge_index, cell_offsets, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()

        # Edge indices (k->j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()
        idx_ji = adj_t_row.storage.row()

        # Remove self-loop triplets d->b->d
        # Check atom as well as cell offset
        cell_offset_kji = cell_offsets[idx_kj] + cell_offsets[idx_ji]
        mask = (idx_i != idx_k) | torch.any(cell_offset_kji != 0, dim=-1)

        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
        idx_kj, idx_ji = idx_kj[mask], idx_ji[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, z, pos, batch=None):
        """ """
        raise NotImplementedError


@registry.register_model("dimenetplusplus_sh")
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
        mesh_conv_layers=3,
        mesh_partition=20,
        mesh_conv_modes=10,
        mesh_channel=32,
        mnum_spherical=3,
        mnum_radial=16,
        num_radial_layer=2,
        radial_embed_size=64,
        radial_hidden_size=128,
        num_sh_gcn_layers=3,
        mesh_hidden_channel=16,
        using_ff=False,
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

        self.mnum_radial = mnum_radial
        self.mnum_spherical = mnum_spherical

        self.num_radial_layer = num_radial_layer
        self.radial_embed_size = radial_embed_size
        self.radial_hidden_size = radial_hidden_size
        self.num_sh_gcn_layers = num_sh_gcn_layers
        self.mesh_hidden_channel = mesh_hidden_channel

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

        # fm stands for "for mesh" (sorry for lazy naming)
        self.embedding_sh = nn.Embedding(100, self.mnum_radial)
        self.irreps_sh = o3.Irreps.spherical_harmonics(self.mnum_spherical, p=1)
        self.irreps_feat = (self.irreps_sh * self.mnum_radial).sort().irreps.simplify()

        self.SphericalGCNs = nn.ModuleList(
            [
                SphericalGCN(
                    (f"{mnum_radial}x0e" if i == 0 else self.irreps_feat),
                    self.irreps_feat,
                    self.irreps_sh,
                    radial_embed_size,
                    num_radial_layer,
                    radial_hidden_size,
                    is_fc=True,
                    **kwargs,
                )
                for i in range(num_sh_gcn_layers)
            ]
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
            **kwargs,
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
            **kwargs,
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

        energy = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)

        edge_index = radius_graph(data.pos, self.cutoff, batch, loop=False)
        src, dst = edge_index
        edge_vec = data.pos[src] - data.pos[dst]
        edge_len = edge_vec.norm(dim=-1) + 1e-8
        edge_feat = o3.spherical_harmonics(
            list(range(self.mnum_spherical + 1)),
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
        z = data.atomic_numbers.long()
        h2 = self.embedding_sh(z)
        for i, gcn in enumerate(self.SphericalGCNs):
            h2 = gcn(edge_index, h2, edge_feat, edge_embed, dim_size=data.pos.size(0))
            if i != self.num_sh_gcn_layers - 1:
                h2 = self.act_sh(h2)

        # 여기서 부터
        mesh = init_particle_mesh(data.cell, self.mesh_partition, self.use_pbc)
        # mesh_feat = init_potential(
        #     data.pos, data.atomic_numbers, data.batch, mesh, self.use_pbc, n=16
        # )
        mesh_feat = torch.zeros(
            mesh["meshgrid_flat"].size(0),
            self.mesh_channel,
            device=data.pos.device,
        )

        mesh_dst, atom_src, edge_vec = radius_and_edge_atom_to_mesh(
            data.pos, data.batch, mesh, self.mesh_cutoff, self.use_pbc
        )

        # atm stands for atom to mesh
        atm_len = torch.norm(edge_vec, dim=-1) + 1e-8
        atm_edge_feat = o3.spherical_harmonics(
            list(range(self.mnum_spherical + 1)),
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
            h2,
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

        h2 = self.mesh_to_atom_gcn.forward(
            (mesh_dst, atom_src),
            mesh_feat,
            atm_edge_feat,
            atm_edge_embed,
            dim_size=data.pos.size(0),
        )

        energy += h2

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
