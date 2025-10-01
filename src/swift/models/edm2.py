# ref : https://github.com/NVlabs/edm2/blob/4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc/training/networks_edm2.py
from typing import Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import ListConfig

from swift.models.abstract import AbstractNetwork

# ----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


# ----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596


# ----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.


def resample(x, f=[1, 1], mode="keep"):
    if mode == "keep":
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = torch.tensor(f, dtype=x.dtype, device=x.device)
    c = x.shape[1]
    if mode == "down":
        return torch.nn.functional.conv2d(
            x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,)
        )
    assert mode == "up"
    return torch.nn.functional.conv_transpose2d(
        x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,)
    )


# ----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t**2)


# ----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).


def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t**2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


# ----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).
# BUT, a smaller bandwidth for better stability (sCM)


class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=0.02):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(torch.float32)
        y = y.outer(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


# ----------------------------------------------------------------------------
# Channel Attention


class ChannelAttention(torch.nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.conv0 = MPConv(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel=[1, 1],
        )
        self.conv1 = MPConv(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel=[1, 1],
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: B×C×H×W → B×C×1×1
        out = self.conv1(mp_silu(self.conv0(self.pool(x))))
        return x * torch.sigmoid(out)


# ----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, pmode="zeros"):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel)
        )
        self.pmode = pmode

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        pad = w.shape[-1] // 2
        if self.pmode == "circular":
            # Only pad left-right (width) with circular, top-bottom with zeros
            x = torch.nn.functional.pad(x, (pad, pad, 0, 0), mode="circular")
            x = torch.nn.functional.pad(x, (0, 0, pad, pad), mode="constant", value=0)
            return torch.nn.functional.conv2d(x, w, padding=0)
        else:
            return torch.nn.functional.conv2d(x, w, padding=pad)


# ----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).


class Block(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        emb_channels,  # Number of embedding channels.
        flavor="enc",  # Flavor: 'enc' or 'dec'.
        resample_mode="keep",  # Resampling: 'keep', 'up', or 'down'.
        resample_filter=[1, 1],  # Resampling filter.
        attention=False,  # Include self-attention?
        channels_per_head=64,  # Number of channels per attention head.
        dropout=0,  # Dropout probability.
        res_balance=0.3,  # Balance between main branch (0) and residual branch (1).
        attn_balance=0.3,  # Balance between main branch (0) and self-attention (1).
        clip_act=256,  # Clip output activations. None = do not clip.
        pmode="zeros",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(
            out_channels if flavor == "enc" else in_channels,
            out_channels,
            kernel=[3, 3],
            pmode=pmode,
        )
        self.emb_linear = MPConv(emb_channels, out_channels * 2, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3, 3], pmode=pmode)
        self.conv_skip = (
            MPConv(in_channels, out_channels, kernel=[1, 1])
            if in_channels != out_channels
            else None
        )
        self.attn_qkv = (
            MPConv(out_channels, out_channels * 3, kernel=[1, 1])
            if self.num_heads != 0
            else None
        )
        self.attn_proj = (
            MPConv(out_channels, out_channels, kernel=[1, 1])
            if self.num_heads != 0
            else None
        )
        # self.ca = ChannelAttention(out_channels) if not attention else None

    def forward(self, x, emb):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)  # pixel-norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain)
        s, b = c.chunk(2, dim=1)
        s = normalize(s.unsqueeze(-1).unsqueeze(-1), dim=1)  # pixel-norm
        b = normalize(b.unsqueeze(-1).unsqueeze(-1), dim=1)  # pixel-norm
        y = mp_silu(y * s + b)
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # if self.ca is not None:
        #     x = self.ca(x)

        # Self-attention.
        # Note: torch.nn.functional.scaled_dot_product_attention() could be used here,
        # but we haven't done sufficient testing to verify that it produces identical results.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3)  # pixel-norm & split
            w = torch.einsum("nhcq,nhck->nhqk", q, k / np.sqrt(q.shape[2])).softmax(
                dim=3
            )
            y = torch.einsum("nhqk,nhck->nhcq", w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x


# ----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).


class UNet(AbstractNetwork):
    def __init__(
        self,
        img_resolution: Tuple[int, int],
        in_channels: int,
        out_channels: int,
        auxiliary_dim: int = 0,  # Class label dimensionality. 0 - no label.
        model_channels: int = 192,  # Base multiplier for the number of channels.
        channel_mult: list[int] = [
            1,
            2,
            3,
            4,
        ],  # Per-resolution multipliers for the number of channels.
        channel_mult_noise: (
            int | None
        ) = None,  # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb: (
            int | None
        ) = None,  # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks: int = 3,  # Number of residual blocks per resolution.
        attn_resolutions: list[Union[int, tuple[int, int], list[int], ListConfig]] = [
            [0, 0]
        ],  # List of resolutions with self-attention.
        label_balance: float = 0.5,  # Balance between noise embedding (0) and class embedding (1).
        concat_balance: float = 0.5,  # Balance between skip connections (0) and main path (1).
        pmode: str = "circular",  # Padding mode for convolutions: 'circular' or 'zeros'.
        **block_kwargs,  # Arguments for Block.
    ):
        super().__init__(img_resolution, in_channels, out_channels)
        self.img_resolution = np.array(img_resolution, dtype=int)
        assert self.img_resolution.shape[0] == 2
        cblock = [model_channels * x for x in channel_mult]
        cnoise = (
            model_channels * channel_mult_noise
            if channel_mult_noise is not None
            else cblock[0]
        )
        cemb = (
            model_channels * channel_mult_emb
            if channel_mult_emb is not None
            else max(cblock)
        )
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = (
            MPConv(auxiliary_dim, cemb, kernel=[]) if auxiliary_dim != 0 else None
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels + 1
        for level, channels in enumerate(cblock):
            res = self.img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"{res[0]}x{res[1]}_conv"] = MPConv(
                    cin, cout, kernel=[3, 3], pmode=pmode
                )
            else:
                self.enc[f"{res[0]}x{res[1]}_down"] = Block(
                    cout,
                    cout,
                    cemb,
                    flavor="enc",
                    resample_mode="down",
                    pmode=pmode,
                    **block_kwargs,
                )
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                attn = res.tolist() in attn_resolutions
                self.enc[f"{res[0]}x{res[1]}_block{idx}"] = Block(
                    cin,
                    cout,
                    cemb,
                    flavor="enc",
                    attention=attn,
                    pmode=pmode,
                    **block_kwargs,
                )

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = self.img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f"{res[0]}x{res[1]}_in0"] = Block(
                    cout,
                    cout,
                    cemb,
                    flavor="dec",
                    attention=True,
                    pmode=pmode,
                    **block_kwargs,
                )
                self.dec[f"{res[0]}x{res[1]}_in1"] = Block(
                    cout,
                    cout,
                    cemb,
                    flavor="dec",
                    pmode=pmode,
                    **block_kwargs,
                )
            else:
                self.dec[f"{res[0]}x{res[1]}_up"] = Block(
                    cout,
                    cout,
                    cemb,
                    flavor="dec",
                    resample_mode="up",
                    pmode=pmode,
                    **block_kwargs,
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                attn = res.tolist() in attn_resolutions
                self.dec[f"{res[0]}x{res[1]}_block{idx}"] = Block(
                    cin,
                    cout,
                    cemb,
                    flavor="dec",
                    attention=attn,
                    pmode=pmode,
                    **block_kwargs,
                )
        self.out_conv = MPConv(cout, out_channels, kernel=[3, 3], pmode=pmode)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        auxiliary: Optional[torch.Tensor] = None,
        return_logvar: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        t = t.flatten()
        assert t.shape[0] == x.shape[0]

        # Embedding.
        emb = self.emb_noise(self.emb_fourier(t))
        if self.emb_label is not None and auxiliary is not None:
            emb = mp_sum(
                emb,
                self.emb_label(auxiliary * np.sqrt(auxiliary.shape[1])),
                t=self.label_balance,
            )
        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if "block" in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    x = torch.randn(2, 69 * 2, 32, 64)
    t = torch.randn(2).reshape(-1, 1)

    model = UNet(
        img_resolution=x.shape[-2:],
        in_channels=x.shape[1],
        out_channels=x.shape[1] // 2,
        auxiliary_dim=0,
        model_channels=192,
        channel_mult=[1, 2, 3, 4],
        attn_resolutions=[[8, 16], [4, 8]],
    )
    summary(model, depth=1)
    y = model(x, t)
    print("Output Shape:", y.shape)
