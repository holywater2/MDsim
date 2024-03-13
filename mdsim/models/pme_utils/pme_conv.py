import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights3 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )
        self.weights4 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                dtype=torch.cfloat,
            )
        )

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class SpectralConv3d_FFNO(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_FFNO, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes1
        self.modes_y = modes2
        self.modes_z = modes3

        self.fourier_weight = nn.ParameterList([])
        for n_modes in [self.modes_x, self.modes_y, self.modes_z]:
            weight = torch.FloatTensor(in_channels, out_channels, n_modes, 2)
            param = nn.Parameter(weight)
            nn.init.xavier_normal_(param)
            self.fourier_weight.append(param)

    def forward(self, x):
        B, I, S1, S2, S3 = x.shape

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ftz = torch.fft.rfftn(x, dim=-1, norm="ortho")
        out_ft = x_ftz.new_zeros(B, I, S1, S2, S3 // 2 + 1)
        out_ft[:, :, :, :, : self.modes_z] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, : self.modes_z],
            torch.view_as_complex(self.fourier_weight[2]),
        )
        xz = torch.fft.irfft(out_ft, n=S3, dim=-1, norm="ortho")

        xz = torch.fft.irfft(out_ft, n=S3, dim=-1, norm="ortho")
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-2, norm="ortho")
        out_ft = x_fty.new_zeros(B, I, S1, S2 // 2 + 1, S3)
        out_ft[:, :, :, : self.modes_y, :] = torch.einsum(
            "bixyz,ioy->boxyz",
            x_fty[:, :, :, : self.modes_y, :],
            torch.view_as_complex(self.fourier_weight[1]),
        )

        xy = torch.fft.irfft(out_ft, n=S2, dim=-2, norm="ortho")

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-3, norm="ortho")
        out_ft = x_ftx.new_zeros(B, I, S1 // 2 + 1, S2, S3)
        out_ft[:, :, : self.modes_x, :, :] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, : self.modes_x, :, :],
            torch.view_as_complex(self.fourier_weight[0]),
        )
        xx = torch.fft.irfft(out_ft, n=S1, dim=-3, norm="ortho")

        # # Combining Dimensions # #
        x = xx + xy + xz

        return x


class PMEConv(nn.Module):
    def __init__(
        self,
        modes1,
        modes2,
        modes3,
        width,
        num_fourier_time,
        padding=0,
        num_layers=2,
        using_ff=False,
    ):
        super(PMEConv, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.num_fourier_time = num_fourier_time
        self.padding = padding  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(self.num_fourier_time + 3, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        self.num_layers = num_layers
        if using_ff:
            self.conv = nn.ModuleList(
                [
                    SpectralConv3d_FFNO(
                        self.width, self.width, self.modes1, self.modes2, self.modes3
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            self.conv = nn.ModuleList(
                [
                    SpectralConv3d(
                        self.width, self.width, self.modes1, self.modes2, self.modes3
                    )
                    for _ in range(self.num_layers)
                ]
            )
        self.w = nn.ModuleList(
            [nn.Conv3d(self.width, self.width, 1) for _ in range(self.num_layers)]
        )
        self.bn = nn.ModuleList(
            [torch.nn.BatchNorm3d(self.width) for _ in range(self.num_layers)]
        )

    def forward(self, residue, fourier_grid):
        x = torch.cat([residue, fourier_grid], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # [B, H, X, Y, Z]
        if self.padding != 0:
            x = F.pad(
                x, [0, self.padding, 0, self.padding, 0, self.padding]
            )  # pad the domain if input is non-periodic

        for i in range(self.num_layers):
            x1 = self.conv[i](x)
            x2 = self.w[i](x)
            x = x1 + x2
            # x = self.bn[i](x)
            if i != self.num_layers - 1:
                x = F.gelu(x)

        if self.padding != 0:
            x = x[..., :, : -self.padding, : -self.padding, : -self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        return x


def get_grid(shape, device):
    batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
    gridx = torch.linspace(0, 1, size_x)
    gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    gridy = torch.linspace(0, 1, size_y)
    gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    gridz = torch.linspace(0, 1, size_z)
    gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
