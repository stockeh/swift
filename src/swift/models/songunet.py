"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

from typing import Optional, Union

import numpy as np
import torch
from omegaconf import ListConfig
from torch.nn.functional import sigmoid, silu

from swift.models.abstract import AbstractNetwork

# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# SE Channel Attention.


class ChannelAttention(torch.nn.Module):
    def __init__(self, in_channels, reduction=16, init=dict()):
        super().__init__()
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel=1,
            bias=False,
            **init,
        )
        self.conv1 = Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel=1,
            bias=False,
            **init,
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: B×C×H×W → B×C×1×1
        out = self.conv1(silu(self.conv0(self.pool(x))))
        return x * sigmoid(out)


# ----------------------------------------------------------------------------
# Fully-connected layer.


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        padding_mode="zeros",
        resample_filter=[1, 1],
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def cylindrical_pad(self, tensor, pad, mode="circular"):
        # Padding for y-direction (Neumann boundary conditions)
        tensor = torch.nn.functional.pad(
            tensor, (0, 0, pad, pad), mode="constant", value=0
        )
        # Padding for x-direction (cylindrical Earth)
        tensor = torch.nn.functional.pad(tensor, (pad, pad, 0, 0), mode=mode)
        return tensor

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                self.cylindrical_pad(x, max(f_pad - w_pad, 0), mode=self.padding_mode),
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
            )
            x = torch.nn.functional.conv2d(
                self.cylindrical_pad(x, w_pad - f_pad, mode=self.padding_mode), w
            )
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(
                self.cylindrical_pad(x, w_pad + f_pad, mode=self.padding_mode), w
            )
            x = torch.nn.functional.conv2d(
                x,
                f.tile([self.out_channels, 1, 1, 1]),
                groups=self.out_channels,
                stride=2,
            )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    self.cylindrical_pad(x, f_pad, mode=self.padding_mode),
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    self.cylindrical_pad(x, f_pad, mode=self.padding_mode),
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                )
            if w is not None:
                x = torch.nn.functional.conv2d(
                    self.cylindrical_pad(x, w_pad, mode=self.padding_mode), w
                )
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


# ----------------------------------------------------------------------------
# Group normalization.


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x


# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(torch.float32),
                (k / np.sqrt(k.shape[1])).to(torch.float32),
            )
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32,
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(
            q.dtype
        ) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(
            k.dtype
        ) / np.sqrt(k.shape[1])
        return dq, dk


# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.


class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        channel_attention=False,
        se_reduction=16,
        num_heads=None,
        channels_per_head=64,
        padding_mode="zeros",
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=[1, 1],
        resample_proj=False,
        adaptive_scale=True,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else (
                num_heads
                if num_heads is not None
                else out_channels // channels_per_head
            )
        )
        self.padding_mode = padding_mode
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            padding_mode=padding_mode,
            resample_filter=resample_filter,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            **init,
        )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=3,
            padding_mode=padding_mode,
            **init_zero,
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                padding_mode=padding_mode,
                resample_filter=resample_filter,
                **init,
            )

        self.ca = (
            ChannelAttention(out_channels, se_reduction, init)
            if channel_attention
            else None
        )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                padding_mode=padding_mode,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                padding_mode=padding_mode,
                **init_zero,
            )

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.ca is not None:
            x = self.ca(x)

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(
                    x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
                .unbind(2)
            )
            w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch


class SongUNet(AbstractNetwork):
    def __init__(
        self,
        img_resolution: Union[int, tuple[int, int], list[int], ListConfig],
        in_channels: int,
        out_channels: int,
        auxiliary_dim: int = 0,  # Number of class auxiliarys, 0 = unconditional.
        augment_dim: int = 0,  # Augmentation auxiliary dimensionality, 0 = no augmentation.
        model_channels: int = 128,  # Base multiplier for num channels.
        channel_mult: Union[list[int], ListConfig] = [
            1,
            2,
            2,
            2,
        ],  # Per-resolution multipliers for num channels.
        channel_mult_emb: int = 4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks: int = 4,  # Number of residual blocks per resolution.
        attn_resolutions: list[Union[int, tuple[int, int], list[int], ListConfig]] = [
            [0, 0]
        ],  # List of resolutions with self-attention.
        dropout: float = 0.10,  # Dropout probability of intermediate activations.
        auxiliary_dropout: float = 0,  # Dropout probability of class auxiliarys for classifier-free guidance.
        eps: float = 1e-6,
        skip_scale: float = np.sqrt(0.5),
        init_mode: str = "xavier_uniform",
        zero_init_weight: float = 1e-5,
        attn_init_weight: float = np.sqrt(0.2),
        padding_mode: str = "circular",  # Default padding mode for convolutions:
        embedding_type: str = "positional",  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type: str = "standard",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type: str = "standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter: Union[int, tuple[int, int], list[int], ListConfig] = [
            1,
            1,
        ],  # Resampling filter: [1,1] DDPM++, [1,3,3,1] NCSN++.
        **kwargs,
    ):
        super().__init__(img_resolution, in_channels, out_channels)
        assert len(self.img_resolution) == 2
        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        assert padding_mode in valid_padding_modes, ValueError(
            f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
        )
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        self.auxiliary_dropout = auxiliary_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        block_kwargs = {
            "emb_channels": emb_channels,
            "num_heads": 1,
            "padding_mode": padding_mode,
            "dropout": dropout,
            "skip_scale": skip_scale,
            "eps": eps,
            "resample_filter": resample_filter,
            "resample_proj": True,
            "adaptive_scale": False,
            "init": {
                "init_mode": init_mode,
            },
            "init_zero": {
                "init_mode": init_mode,
                "init_weight": zero_init_weight,
            },
            "init_attn": {
                "init_mode": init_mode,
                "init_weight": attn_init_weight,
            },
        }

        # Mapping.
        self.map_noise = (
            PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            if embedding_type == "positional"
            else FourierEmbedding(num_channels=noise_channels)
        )
        self.map_auxiliary = (
            Linear(
                in_features=auxiliary_dim,
                out_features=noise_channels,
                init_mode=init_mode,
            )
            if auxiliary_dim
            else None
        )
        self.map_augment = (
            Linear(
                in_features=augment_dim,
                out_features=noise_channels,
                bias=False,
                init_mode=init_mode,
            )
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(
            in_features=noise_channels,
            out_features=emb_channels,
            init_mode=init_mode,
        )
        self.map_layer1 = Linear(
            in_features=emb_channels,
            out_features=emb_channels,
            init_mode=init_mode,
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            # NOTE: the following is equivalent to:
            # res = res // (2 ** level)
            res = np.array(self.img_resolution) >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res[0]}x{res[1]}_conv"] = Conv2d(
                    in_channels=cin,
                    out_channels=cout,
                    kernel=3,
                    padding_mode=padding_mode,
                    init_mode=init_mode,
                )
            else:
                self.enc[f"{res[0]}x{res[1]}_down"] = UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    down=True,
                    channel_attention=True,
                    **block_kwargs,
                )
                if encoder_type == "skip":
                    self.enc[f"{res[0]}x{res[1]}_aux_down"] = Conv2d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        padding_mode=padding_mode,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res[0]}x{res[1]}_aux_skip"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=1,
                        padding_mode=padding_mode,
                        init_mode=init_mode,
                    )
                if encoder_type == "residual":
                    self.enc[f"{res[0]}x{res[1]}_aux_residual"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        padding_mode=padding_mode,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        init_mode=init_mode,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res.tolist() in attn_resolutions
                self.enc[f"{res[0]}x{res[1]}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=attn,
                    channel_attention=True,
                    **block_kwargs,
                )
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            # NOTE: the following is equivalent to:
            # res = res // (2 ** level)
            res = np.array(img_resolution) >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res[0]}x{res[1]}_in0"] = UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    attention=True,
                    channel_attention=True,
                    **block_kwargs,
                )
                self.dec[f"{res[0]}x{res[1]}_in1"] = UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    attention=False,
                    channel_attention=True,
                    **block_kwargs,
                )
            else:
                self.dec[f"{res[0]}x{res[1]}_up"] = UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    up=True,
                    channel_attention=True,
                    **block_kwargs,
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res.tolist() in attn_resolutions
                self.dec[f"{res[0]}x{res[1]}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=attn,
                    channel_attention=True,
                    **block_kwargs,
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res[0]}x{res[1]}_aux_up"] = Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        padding_mode=padding_mode,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res[0]}x{res[1]}_aux_norm"] = GroupNorm(
                    num_channels=cout, eps=1e-6
                )
                self.dec[f"{res[0]}x{res[1]}_aux_conv"] = Conv2d(
                    in_channels=cout,
                    out_channels=out_channels,
                    kernel=3,
                    padding_mode=padding_mode,
                    **block_kwargs["init_zero"],
                )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        auxiliary: Optional[torch.Tensor] = None,
        augment_auxiliarys: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if t.dim() == 0:  # t: noise, timesteps, etc.
            t = t.repeat(x.shape[0])
        # Mapping.
        emb = self.map_noise(t)
        emb = (
            emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        )  # swap sin/cos
        if self.map_auxiliary is not None and auxiliary is not None:
            tmp = auxiliary
            if self.training and self.auxiliary_dropout:
                tmp = tmp * (
                    torch.rand([x.shape[0], 1], device=x.device)
                    >= self.auxiliary_dropout
                ).to(tmp.dtype)
            # TODO: issue with large # in_features?
            emb = emb + self.map_auxiliary(
                tmp * np.sqrt(self.map_auxiliary.in_features)
            )
        if self.map_augment is not None and augment_auxiliarys is not None:
            emb = emb + self.map_augment(augment_auxiliarys)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)

        return aux


if __name__ == "__main__":
    from torchinfo import summary

    x = torch.randn(32, 69 * 2, 128, 256)

    model = SongUNet(
        img_resolution=x.shape[2:],
        in_channels=x.shape[1],
        out_channels=69,
        model_channels=256,
        attn_resolutions=[[64, 128], [32, 64], [16, 32]],
        channel_mult_noise=1,
        resample_filter=[1, 1],
        channel_mult=[2, 2, 2, 4],
    )
    summary(model, depth=1)

    t = torch.randn(x.shape[0])
    # t = torch.tensor([0.0])
    # _ = model(x, t)
