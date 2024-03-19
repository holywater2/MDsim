import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_scatter import scatter
from torch_geometric.nn import radius_graph, radius
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Extract, Activation
from e3nn.math import soft_one_hot_linspace

from mdsim.models.pme_utils.utils import Rcovalent

# covalent radii of elements
rcov = Rcovalent


class ScalarActivation(nn.Module):
    """
    Use the invariant scalar features to gate higher order equivariant features.
    Adapted from `e3nn.nn.Gate`.
    """

    def __init__(self, irreps_in, act_scalars, act_gates):
        """
        :param irreps_in: input representations
        :param act_scalars: scalar activation function
        :param act_gates: gate activation function (for higher order features)
        """
        super(ScalarActivation, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.num_spherical = len(self.irreps_in)

        irreps_scalars = self.irreps_in[0:1]
        irreps_gates = irreps_scalars * (self.num_spherical - 1)
        irreps_gated = self.irreps_in[1:]
        self.act_scalars = Activation(irreps_scalars, [act_scalars])
        self.act_gates = Activation(
            irreps_gates, [act_gates] * (self.num_spherical - 1)
        )
        self.extract = Extract(
            self.irreps_in,
            [irreps_scalars, irreps_gated],
            instructions=[(0,), tuple(range(1, self.irreps_in.lmax + 1))],
        )
        self.mul = o3.ElementwiseTensorProduct(irreps_gates, irreps_gated)

    def forward(self, features):
        scalars, gated = self.extract(features)
        scalars_out = self.act_scalars(scalars)
        if gated.shape[-1]:
            gates = self.act_gates(scalars.repeat(1, self.num_spherical - 1))
            gated_out = self.mul(gates, gated)
            features = torch.cat([scalars_out, gated_out], dim=-1)
        else:
            features = scalars_out
        return features


class NormActivation(nn.Module):
    """
    Use the norm of the higher order equivariant features to gate themselves.
    Idea from the TFN paper.
    """

    def __init__(
        self, irreps_in, act_scalars=torch.nn.functional.silu, act_vectors=torch.sigmoid
    ):
        """
        :param irreps_in: input representations
        :param act_scalars: scalar activation function
        :param act_vectors: vector activation function (for the norm of higher order features)
        """
        super(NormActivation, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.scalar_irreps = self.irreps_in[0:1]
        self.vector_irreps = self.irreps_in[1:]
        self.act_scalars = act_scalars
        self.act_vectors = act_vectors
        self.scalar_idx = self.irreps_in[0].mul

        inner_out = o3.Irreps([(mul, (0, 1)) for mul, _ in self.vector_irreps])
        self.inner_prod = o3.TensorProduct(
            self.vector_irreps,
            self.vector_irreps,
            inner_out,
            [(i, i, i, "uuu", False) for i in range(len(self.vector_irreps))],
        )
        self.mul = o3.ElementwiseTensorProduct(inner_out, self.vector_irreps)

    def forward(self, features):
        scalars = self.act_scalars(features[..., : self.scalar_idx])
        vectors = features[..., self.scalar_idx :]
        norm = torch.sqrt(self.inner_prod(vectors, vectors) + 1e-8)
        act = self.act_vectors(norm)
        vectors_out = self.mul(act, vectors)
        return torch.cat([scalars, vectors_out], dim=-1)


class GCNLayer(nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_edge,
        radial_embed_size,
        num_radial_layer,
        radial_hidden_size,
        is_fc=True,
        use_sc=True,
        irrep_normalization="component",
        path_normalization="element",
        *args,
        **kwargs,
    ):
        r"""
        .. math::
            z_w=\sum_{uv}w_{uvw}x_u\otimes y_v=\sum_{u}w_{uw}x_u \otimes y

        Else, we have

        .. math::
            z_u=x_u\otimes \sum_v w_{uv}y_v=w_u (x_u\otimes y)

        Here, uvw are radial (channel) indices of the first input, second input, and output, respectively.
        Notice that in our model, the second input is always the spherical harmonics of the edge vector,
        so the index v can be safely ignored.

        :param irreps_in: irreducible representations of input node features
        :param irreps_out: irreducible representations of output node features
        :param irreps_edge: irreducible representations of edge features
        :param radial_embed_size: embedding size of the edge length
        :param num_radial_layer: number of hidden layers in the radial network
        :param radial_hidden_size: hidden size of the radial network
        :param is_fc: whether to use fully connected tensor product
        :param use_sc: whether to use self-connection
        :param irrep_normalization: representation normalization passed to the `o3.FullyConnectedTensorProduct`
        :param path_normalization: path normalization passed to the `o3.FullyConnectedTensorProduct`
        """
        super(GCNLayer, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.radial_embed_size = radial_embed_size
        self.num_radial_layer = num_radial_layer
        self.radial_hidden_size = radial_hidden_size
        self.is_fc = is_fc
        self.use_sc = use_sc

        if self.is_fc:
            self.tp = o3.FullyConnectedTensorProduct(
                self.irreps_in,
                self.irreps_edge,
                self.irreps_out,
                internal_weights=False,
                shared_weights=False,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
            )
        else:
            instr = [
                (i_1, i_2, i_out, "uvu", True)
                for i_1, (_, ir_1) in enumerate(self.irreps_in)
                for i_2, (_, ir_edge) in enumerate(self.irreps_edge)
                for i_out, (_, ir_out) in enumerate(self.irreps_out)
                if ir_out in ir_1 * ir_edge
            ]
            # import pdb; pdb.set_trace()
            self.tp = o3.TensorProduct(
                self.irreps_in,
                self.irreps_edge,
                self.irreps_out,
                instr,
                internal_weights=False,
                shared_weights=False,
                irrep_normalization=irrep_normalization,
                path_normalization=path_normalization,
            )
        self.fc = FullyConnectedNet(
            [radial_embed_size]
            + num_radial_layer * [radial_hidden_size]
            + [self.tp.weight_numel],
            torch.nn.functional.silu,
        )
        self.sc = None
        if self.use_sc:
            self.sc = o3.Linear(self.irreps_in, self.irreps_out)

    def forward(self, edge_index, node_feat, edge_feat, edge_embed, dim_size=None):
        src, dst = edge_index
        weight = self.fc(edge_embed)
        out = self.tp(node_feat[src], edge_feat, weight=weight)
        out = scatter(out, dst, dim=0, dim_size=dim_size, reduce="sum")
        if self.use_sc:
            out = out + self.sc(node_feat)
        return out

    def forward_mean(self, edge_index, node_feat, edge_feat, edge_embed, dim_size=None):
        src, dst = edge_index
        weight = self.fc(edge_embed)
        out = self.tp(node_feat[src], edge_feat, weight=weight)
        out = scatter(out, dst, dim=0, dim_size=dim_size, reduce="mean")
        if self.use_sc:
            out = out + self.sc(node_feat)
        return out


def init_particle_mesh(cell, n_partition, pbc=False):
    """generate particle mesh for given cell and n_partition.

    Args:
        cell : (num_batch, 3, 3) the cell vectors of the data.
        n_partition : the number of partition in each axis. (ex. 10 implies 10
                    grid points in each axis, total 10^3 grid points in 3D)
        pbc (bool, optional): Specify if the system is periodic. Defaults to False.

    Returns:
        res: dictation of the particle mesh information
            meshgrid: (num_batch, n_partition, n_partition, n_partition, 3) the meshgrid
                       of the particle mesh. Each grid point is represented by the
                       coordinate in the real space. 2,3,4 axis are the discretized index
                       of the particle mesh. (n,2,3,4,:) imples coordinate of n-th grid point.
                       last axis is the coordinate of the grid point.
            meshgrid_flat: (num_batch*n_partition^3, 3) the flattened meshgrid.
            meshgrid_batch: (num_batch*n_partition^3) the batch index of the meshgrid points.
                            used for message passing

        when pbc is True, the following information is also returned.
            super_meshgrid: (num_batch, super_n_partition, super_n_partition, super_n_partition, 3)
                            the meshgrid of the super particle mesh. used for pbc. In the pbc
                            situation, the particle mesh is 2x larger than the original particle to
                            consider the periodic boundary condition.
                            (20 grid points in each axis, then middle 10 grid points are the original
                            and the rest 5 grid points are the periodic images of the original 10 grid points)
            super_meshgrid_flat: (num_batch*super_n_partition^3, 3) the flattened super meshgrid.
            super_meshgrid_batch: (num_batch*super_n_partition^3) the batch index of the super meshgrid points.
            super_meshgrid_flat_idx: (super_n_partition^3) the corresponding index of the super meshgrid points in the
                                original particle mesh. used for message passing.
            super_meshgrid_idx_to_batch_idx: (num_batch*super_n_partition^3) the index of the super meshgrid points
                                Used for map the result of radius which is in the super meshgrid to the original

    Note that the tensor shape of the meshgrid is (num_batch, n_partition, n_partition, n_partition, 3) and
        2,3,4 axis are the discretized index of the particle mesh.

    """
    cell = cell.detach()
    num_batch = cell.shape[0]
    device = cell.device

    grid_frac_axis = torch.linspace(0, 1, n_partition)

    # for equivariance
    grid_frac_axis -= 0.5

    grid_frac = torch.meshgrid(grid_frac_axis, grid_frac_axis, grid_frac_axis)
    grid_frac = torch.stack(grid_frac, dim=-1).to(device)

    # b: batch
    # x,y,z: grid_frac_index
    # l: 3 = (v_1,v_2,v_3)
    # m: 3 = (x,y,Z)
    meshgrid = torch.einsum("xyzl,blm->bxyzm", grid_frac, cell).to(device)
    # meshgrid_flat = meshgrid.reshape(num_batch, -1, 3).to(device)
    meshgrid_flat = meshgrid.reshape(-1, 3).to(device)
    meshgrid_batch = (
        torch.arange(num_batch).repeat_interleave(n_partition**3).to(device)
    )
    mesh_length = torch.norm(cell, dim=-1) / n_partition

    res = {
        "cell": cell,
        "n_partition": n_partition,
        "meshgrid": meshgrid.detach(),
        "meshgrid_flat": meshgrid_flat.detach(),
        "meshgrid_batch": meshgrid_batch,
        "pbc": pbc,
        "mesh_length": mesh_length.detach(),
    }

    if pbc:
        n_partition_half = len(grid_frac_axis) // 2

        # generate 2x larger grid
        super_grid_frac_axis = torch.cat(
            [
                grid_frac_axis[n_partition_half:-1] - 1,
                grid_frac_axis,
                grid_frac_axis[1:n_partition_half] + 1,
            ]
        )
        super_n_partition = len(super_grid_frac_axis)

        super_grid_frac = torch.meshgrid(
            super_grid_frac_axis, super_grid_frac_axis, super_grid_frac_axis
        )
        super_meshgrid = torch.stack(super_grid_frac, dim=-1).to(device)

        # Mapping super index to normal index
        partition_idx = torch.arange(len(grid_frac_axis)).to(device)
        # last index corresponds to the first index (Mirror Image)
        partition_idx[-1] = 0

        # (3 4 5 / 0 1 2 3 4 5 /0 1 2)
        super_partition_idx = torch.cat(
            [
                partition_idx[n_partition_half:-1],
                partition_idx,
                partition_idx[1:n_partition_half],
            ]
        ).to(device)
        normal_idx_wo_last = (
            torch.arange((len(partition_idx) - 1) ** 3)
            .reshape(
                len(partition_idx) - 1, len(partition_idx) - 1, len(partition_idx) - 1
            )
            .to(device)
        )
        super_meshgrid_idx_helper = (
            torch.stack(
                torch.meshgrid(
                    super_partition_idx, super_partition_idx, super_partition_idx
                ),
                dim=-1,
            )
            .reshape(-1, 3)
            .to(device)
        )
        # Representing (v_1,v_2,v_3) index of the super meshgrid
        # (3,4,5,:) represents 3rd, 4th, 5th grid points in each axis and its coordinate
        # corresponding to the original particle mesh grid index

        # Representing the node index of the super meshgrid in the original particle mesh
        super_meshgrid_flat_to_normal_idx = (
            normal_idx_wo_last[
                super_meshgrid_idx_helper[:, 0],
                super_meshgrid_idx_helper[:, 1],
                super_meshgrid_idx_helper[:, 2],
            ]
            .reshape(-1)
            .to(device)
        )

        # Convert the fractional index to the real coordinate by matrix multiplication with cell
        super_meshgrid = torch.einsum("xyzl,blm->bxyzm", super_meshgrid, cell).to(
            device
        )
        super_meshgrid_flat = super_meshgrid.reshape(-1, 3)
        super_meshgrid_batch = torch.arange(num_batch, device=device).repeat_interleave(
            len(super_partition_idx) ** 3
        )

        super_meshgrid_flat_idx_to_batch = torch.arange(
            num_batch, device=device
        ).repeat_interleave(
            len(super_meshgrid_flat_to_normal_idx)
        ) * n_partition**3 + super_meshgrid_flat_to_normal_idx.repeat(
            num_batch
        )

        res.update(
            {
                "super_n_partition": super_n_partition,
                "super_meshgrid": super_meshgrid.detach(),
                "super_meshgrid_flat": super_meshgrid_flat.detach(),
                "super_meshgrid_batch": super_meshgrid_batch.detach(),
                "super_meshgrid_flat_to_normal_idx": super_meshgrid_flat_to_normal_idx,
                "super_meshgrid_flat_idx_to_batch": super_meshgrid_flat_idx_to_batch,
            }
        )
    return res


def init_potential(
    atom_coord,
    atomic_number,
    batch,
    meshgrid,
    pbc=False,
    gaussian_bins_start=0.5,
    gaussian_bins_end=5.0,
    gaussian_bins_number=16,
):
    num_batch = meshgrid["cell"].size(0)
    nd = meshgrid["meshgrid"].size(1)
    minimal_dist_grid = torch.zeros(num_batch, nd * nd * nd).to(atom_coord.device)
    pot_feat = torch.zeros(num_batch, nd, nd, nd, gaussian_bins_number).to(
        atom_coord.device
    )
    Rcovalent = rcov.to(atom_coord.device)

    # gaussian_bins
    gb = (
        torch.linspace(gaussian_bins_start, gaussian_bins_end, gaussian_bins_number)
        .reshape(1, -1, 1, 1)
        .to(atom_coord.device)
    )
    if pbc:
        for batch_idx in range(num_batch):
            bmesh = meshgrid["meshgrid"][batch_idx].reshape(
                -1, 1, 1, 1, 3
            )  # (nd^3, 1,1,1,3)
            bcoord = atom_coord[batch == batch_idx].reshape(
                1, 1, -1, 1, 3
            )  # (1,1,n,1,3)
            bcell = meshgrid["cell"][batch_idx]  # (3, 3)
            super_bcell = torch.stack(
                [
                    i * bcell[0] + j * bcell[1] + k * bcell[2]
                    for i in [-1, 0, 1]
                    for j in [-1, 0, 1]
                    for k in [-1, 0, 1]
                ],
                dim=0,
            ).reshape(1, 1, 1, 27, 3)
            all_norm = torch.norm(
                bmesh - bcoord + super_bcell, dim=-1
            )  # (nd^3, n, 27, 1)
            atom_dist = Rcovalent[atomic_number[batch == batch_idx]].reshape(
                1, 1, -1, 1
            )
            exp = torch.exp(-gb * (all_norm**2) / atom_dist)
            res = torch.sum(exp, dim=[-1, -2]).reshape(nd, nd, nd, -1)
            pot_feat[batch_idx] = res
    else:
        raise NotImplementedError("Not implemented for non-PBC case")

    pot_feat = pot_feat.reshape(-1, gaussian_bins_number)
    return pot_feat


def init_feat(
    atom_coord,
    atomic_number,
    batch,
    meshgrid,
    pbc=False,
    feat_type="zeros",
    n=16,
    gaussian_bins_start=0.5,
    gaussian_bins_end=5.0,
):
    assert feat_type in ["zeros", "potential"]
    assert atom_coord.size(0) == batch.size(0)
    assert atomic_number.size(0) == batch.size(0)
    assert n > 0

    if feat_type == "zeros":
        mesh_feat = torch.zeros(meshgrid["meshgrid_flat"].size(0), n)
    elif feat_type == "potential":
        mesh_feat = init_potential(
            atom_coord,
            atomic_number,
            batch,
            meshgrid,
            pbc,
            gaussian_bins_start,
            gaussian_bins_end,
            n,
        )

    return mesh_feat.to(atom_coord.device)


def radius_and_edge_atom_to_mesh(atom_coord, batch, particle_mesh, cutoff, pbc=False):
    """Calculate the source and destination within radius.
        It is used for message passing

    Args:
        atom_coord (N, 3):  the coordinate of the atoms
        batch (N): the batch index of the atoms
        particle_mesh (dict): the particle mesh information generated by init_particle_mesh
        cutoff (float): the cutoff radius for the message passing
        pbc (bool, optional): Periodic Boundary Condition. Defaults to False.

    Returns:
        tuple: mesh_dst, atom_src, edge_vec
            mesh_dst (batch*n_partition^3): the destination of the message passing
            atom_src (batch): the source of the message passing
            edge_vec (batch, 3): the edge vector from the source to the destination
    """
    assert cutoff > 0
    assert atom_coord.size(0) == batch.size(0)

    # Calculate the source and destination within radius.
    # It is used for message passing
    # dst: particle mesh destination
    # src: atom source
    # src -> dst : atom -> particle mesh
    if pbc:
        super_mesh_dst, atom_src = radius(
            atom_coord,
            particle_mesh["super_meshgrid_flat"],
            cutoff,
            batch,
            particle_mesh["super_meshgrid_batch"],
        )
        mesh_dst = particle_mesh["super_meshgrid_flat_idx_to_batch"][super_mesh_dst]

        # edge vector calculated in the supercell
        mesh_pos = particle_mesh["super_meshgrid_flat"][super_mesh_dst]
        edge_vec = (
            particle_mesh["super_meshgrid_flat"][super_mesh_dst] - atom_coord[atom_src]
        )
    else:
        mesh_dst, atom_src = radius(
            atom_coord,
            particle_mesh["particle_mesh_flat"],
            cutoff,
            batch,
            particle_mesh["particle_mesh_batch"],
        )
        mesh_pos = particle_mesh["super_meshgrid_flat"][super_mesh_dst]
        edge_vec = particle_mesh["particle_mesh_flat"][mesh_dst] - atom_coord[atom_src]
    return mesh_dst, atom_src, edge_vec, mesh_pos


def process_atom_to_mesh(atom_coord, batch, particle_mesh, cutoff, pbc=False):
    mesh_dst, atom_src, edge_vec, mesh_pos = radius_and_edge_atom_to_mesh(
        atom_coord, batch, particle_mesh, cutoff, pbc
    )
    atom_with_mesh = torch.cat([atom_coord, particle_mesh["meshgrid_flat"]], dim=0)
    atom_with_super_mesh = torch.cat([atom_coord, mesh_pos], dim=0)
    atom_with_mesh_batch = torch.cat([batch, particle_mesh["meshgrid_batch"]], dim=0)

    atom_with_mesh_dst = mesh_dst + atom_coord.size(0)
    atom_with_mesh_src = atom_src

    return (
        atom_with_mesh,
        atom_with_super_mesh,
        atom_with_mesh_batch,
        atom_with_mesh_dst,
        atom_with_mesh_src,
        edge_vec,
    )


# def mesh_embedding(atom_coord, batch, particle_mesh, cutoff, options=None):
#     layer_style = "Tensor Field Network based"

#     pbc = options["pbc"]
#     num_spherical = options["num_spherical"]
#     radial_embed_size = options["radial_embed_size"]

#     mesh_dst, atom_src, atm_edge = radius_and_edge_atom_to_mesh(
#         atom_coord, batch, particle_mesh, cutoff, pbc
#     )

#     # atm : from atom to mesh
#     atm_norm = torch.norm(atm_edge, dim=-1) + 1e-8
#     atm_edge_feat = o3.spherical_harmonics(
#         list(range(num_spherical + 1)),
#         atm_edge / (atm_norm[..., None] + 1e-8),
#         normalize=False,
#         normalization="integral",
#     )
#     atm_edge_embed = soft_one_hot_linspace(
#         atm_norm,
#         start=0.0,
#         end=cutoff,
#         number=radial_embed_size,
#         basis="gaussian",
#         cutoff=False,
#     ).mul(radial_embed_size**0.5)

#     # feat: (probe_src.size(0), sp_feat(4-> 400))
#     # probe_feat: (b,f*f*f,c)

#     mesh_feat = probe_gcn.

#     probe_feat = self.probe_gcn.forward_mean(
#         (probe_src, probe_dst),
#         feat,
#         probe_edge_feat,
#         probe_edge_embed,
#         dim_size=probe_flat.size(0),
#     )

#     probe_feat = self.act_probe(probe_feat)

#     # probe_feat: (b,f,f,f,c)
#     probe_feat = probe_feat.reshape(
#         grid.size(0), self.num_fourier, self.num_fourier, self.num_fourier, -1
#     )
