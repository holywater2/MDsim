import torch
import torch.nn as nn
from torch_scatter import scatter

import math

from e3nn import o3
from e3nn.nn import FullyConnectedNet, Extract, Activation


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


class SphericalGCN(nn.Module):
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
        If the tensor product is fully connected, we have (for every path)

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
        super(SphericalGCN, self).__init__()
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

    def forward(
        self, edge_index, node_feat, edge_feat, edge_embed, dim_size=None, reduce="mean"
    ):
        assert reduce in ["sum", "mean"]
        src, dst = edge_index
        weight = self.fc(edge_embed)
        out = self.tp(node_feat[src], edge_feat, weight=weight)
        out = scatter(out, dst, dim=0, dim_size=dim_size, reduce=reduce)
        if self.use_sc:
            out = out + self.sc(node_feat)
        return out


class BroadcastGTOTensor(nn.Module):
    r"""
    Broadcast between spherical tensors of the Gaussian Type Orbitals (GTOs):

    .. math::
        \{a_{clm}, 1\le c\le c_{max}, 0\le\ell\le\ell_{max}, -\ell\le m\le\ell\}

    For efficiency reason, the feature tensor is indexed by l, c, m.
    For example, for lmax = 3, cmax = 2, we have a tensor of 1s2s 1p2p 1d2d 1f2f.
    Currently, we support the following broadcasting:
    lc -> lcm;
    m -> lcm.
    """

    def __init__(self, lmax, cmax, src="lc", dst="lcm"):
        super(BroadcastGTOTensor, self).__init__()
        assert src in ["lc", "m"]
        assert dst in ["lcm"]
        self.src = src
        self.dst = dst
        self.lmax = lmax
        self.cmax = cmax

        if src == "lc":
            self.src_dim = (lmax + 1) * cmax
        else:
            self.src_dim = (lmax + 1) ** 2
        self.dst_dim = (lmax + 1) ** 2 * cmax

        if src == "lc":
            indices = self._generate_lc2lcm_indices()
        else:
            indices = self._generate_m2lcm_indices()
        self.register_buffer("indices", indices)

    def _generate_lc2lcm_indices(self):
        r"""
        lc -> lcm
        .. math::
            1s2s 1p2p → 1s2s 1p_x1p_y1p_z2p_x2p_y2p_z
        [0, 1, 2, 2, 2, 3, 3, 3]

        :return: (lmax+1)^2 * cmax
        """
        indices = [
            l * self.cmax + c
            for l in range(self.lmax + 1)
            for c in range(self.cmax)
            for _ in range(2 * l + 1)
        ]
        return torch.LongTensor(indices)

    def _generate_m2lcm_indices(self):
        r"""
        m -> lcm
        .. math::
            s p_x p_y p_z → 1s2s 1p_x1p_y1p_z2p_x2p_y2p_z
        [0, 0, 1, 2, 3, 1, 2, 3]

        :return: (lmax+1)^2 * cmax
        """
        indices = [
            l * l + m
            for l in range(self.lmax + 1)
            for _ in range(self.cmax)
            for m in range(2 * l + 1)
        ]
        return torch.LongTensor(indices)

    def forward(self, x):
        """
        Apply broadcasting to x.
        :param x: (..., src_dim)
        :return: (..., dst_dim)
        """
        assert x.size(-1) == self.src_dim, (
            f"Input dimension mismatch! "
            f"Should be {self.src_dim}, but got {x.size(-1)} instead!"
        )
        if self.src == self.dst:
            return x
        return x[..., self.indices]


class GaussianOrbital(nn.Module):
    r"""
    Gaussian-type orbital

    .. math::
        \psi_{n\ell m}(\mathbf{r})=\sqrt{\frac{2(2a_n)^{\ell+3/2}}{\Gamma(\ell+3/2)}}
        \exp(-a_n r^2) r^\ell Y_{\ell}^m(\hat{\mathbf{r}})

    """

    def __init__(self, gauss_start, gauss_end, num_gauss, lmax=7):
        super(GaussianOrbital, self).__init__()
        self.gauss_start = gauss_start
        self.gauss_end = gauss_end
        self.num_gauss = num_gauss
        self.lmax = lmax

        self.lc2lcm = BroadcastGTOTensor(lmax, num_gauss, src="lc", dst="lcm")
        self.m2lcm = BroadcastGTOTensor(lmax, num_gauss, src="m", dst="lcm")
        self.gauss: torch.Tensor
        self.lognorm: torch.Tensor

        self.register_buffer("gauss", torch.linspace(gauss_start, gauss_end, num_gauss))
        self.register_buffer("lognorm", self._generate_lognorm())

    def _generate_lognorm(self):
        power = (torch.arange(self.lmax + 1) + 1.5).unsqueeze(-1)  # (l, 1)
        numerator = power * torch.log(2 * self.gauss).unsqueeze(0) + math.log(
            2
        )  # (l, c)
        denominator = torch.special.gammaln(power)
        lognorm = (numerator - denominator) / 2
        return lognorm.view(-1)  # (l * c)

    def forward(self, vec):
        """
        Evaluate the basis functions
        :param vec: un-normalized vectors of (..., 3)
        :return: basis values of (..., (l+1)^2 * c)
        """
        # spherical
        device = vec.device
        r = vec.norm(dim=-1) + 1e-8
        spherical = o3.spherical_harmonics(
            list(range(self.lmax + 1)),
            vec / r[..., None],
            normalize=False,
            normalization="integral",
        )

        # radial
        r = r.unsqueeze(-1)
        lognorm = self.lognorm * torch.ones_like(r)  # (..., l * c)
        exponent = -self.gauss * (r * r)  # (..., c)
        poly = torch.arange(
            self.lmax + 1, dtype=torch.float, device=device
        ) * torch.log(
            r
        )  # (..., l)
        log = exponent.unsqueeze(-2) + poly.unsqueeze(-1)  # (..., l, c)
        radial = torch.exp(log.view(*log.size()[:-2], -1) + lognorm)  # (..., l * c)
        return self.lc2lcm(radial) * self.m2lcm(spherical)  # (..., (l+1)^2 * c)
