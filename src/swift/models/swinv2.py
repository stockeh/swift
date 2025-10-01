import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from omegaconf import ListConfig

from swift.models.abstract import AbstractNetwork, Shape2D

# ----------------------------------------------------------------------------
# Utility Functions


def window_partition(x: torch.Tensor, window_size: tuple[int, int]):
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)"""
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], C)
    )
    return windows


def window_reverse(
    windows: torch.Tensor, window_size: tuple[int, int], img_size: tuple[int, int]
):
    """(num_windows * B, window_size[0], window_size[1], C) -> (B, H, W, C)"""
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(
        -1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000):
    """Sinusoidal timestep embeddings."""
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=t.dtype) / half
    ).to(device=t.device)
    args = t[:, None].to(t.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    embedding = (
        embedding.reshape(embedding.shape[0], 2, -1).flip(1).reshape(*embedding.shape)
    )  # flip sin/cos as done with edm

    return embedding


# ----------------------------------------------------------------------------
# Swin Modules


class LatentEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l1 = nn.Linear(dim, dim, bias=True)
        self.l2 = nn.Linear(dim, dim, bias=True)

    def forward(self, emb):
        return F.silu(self.l2(F.silu(self.l1(emb))))


class ModulatedNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps)
        self.modulation = nn.Linear(dim, dim * 2, bias=True)

    def forward(self, x, t):
        x = self.norm(x)  # b, n, d
        scale, shift = self.modulation(t).chunk(2, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    """SwiGLU FeedForward"""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = ModulatedNorm(dim)
        self.w1 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, t):
        gate, up_proj = self.w1(x).chunk(2, dim=-1)
        x = self.w2(F.silu(gate) * up_proj)
        x = self.norm(x, t)  # new: post-norm
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, flash=True):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.flash = flash
        self.norm = ModulatedNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)

        self.scale = nn.Parameter(torch.log(10 * torch.ones(1, heads, 1, 1)))

    def forward(self, x, t, jvp: bool = False):
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b n (h d) -> b h n d", h=self.heads)
        q, k, v = qkv.chunk(3, dim=-1)

        q = (
            F.normalize(q, dim=-1)
            * torch.clamp(self.scale, max=math.log(1.0 / 0.01)).exp()
        )
        k = F.normalize(k, dim=-1)

        if self.flash and not jvp:
            x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        else:
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.wo(x)
        x = self.norm(x, t)  # new: post-norm
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        depth,
        dim,
        heads,
        window_size,
        grid_size,
        shift_size,
        flash,
    ):
        super().__init__()

        self.window_size = window_size
        self.grid_size = grid_size
        self.shift_size = shift_size

        head_dim = dim // heads
        mlp_dim = int(8 / 3.0 * dim)

        self.layers = nn.Sequential(
            *[
                nn.ModuleList(
                    [
                        Attention(dim, heads, head_dim, flash),
                        FeedForward(dim, mlp_dim),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, jvp: bool = False
    ) -> torch.Tensor:
        sh, sw = self.shift_size
        do_shift: bool = any(self.shift_size)

        # expand t to match the number of windows
        repeat_factor = (self.grid_size[0] // self.window_size[0]) * (
            self.grid_size[1] // self.window_size[1]
        )
        t_expanded = t.repeat_interleave(repeat_factor, dim=0)  # num_windows * b, d

        for i, (attn, ff) in enumerate(self.layers):  # type:ignore  ??
            xp = x

            x = x.view(-1, self.grid_size[0], self.grid_size[1], x.shape[-1])
            _, h, w, d = x.shape

            # cyclic shift
            if do_shift and i % 2 != 0:
                x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))

            # partition windows
            x = window_partition(x, self.window_size)
            x = x.view(-1, self.window_size[0] * self.window_size[1], d)

            x = attn(x, t_expanded, jvp)  # num_windows * b, n, d

            # merge windows
            x = x.view(-1, self.window_size[0], self.window_size[1], d)
            x = window_reverse(x, self.window_size, (h, w))

            # reverse cyclic shift
            if do_shift and i % 2 != 0:
                x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))
            x = x.view(-1, h * w, d)

            x = xp + x
            x = x + ff(x, t)

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, dim):
        super().__init__()
        self.patch_size = p1, p2 = patch_size
        self.emb = nn.Linear(in_channels * p1 * p2, dim)

    def forward(self, x):
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
        )
        return self.emb(x)


class OutputHead(nn.Module):
    def __init__(self, dim, out_channels, patch_size, grid_size):
        super().__init__()
        p1, p2 = patch_size
        gh, gw = grid_size

        self.head = nn.Sequential(
            nn.Linear(dim, out_channels * p1 * p2, bias=False),  # b, n, c*p1*p2
            Rearrange(
                "b (h w) (c p1 p2) -> b c (h p1) (w p2)", p1=p1, p2=p2, h=gh, w=gw
            ),
        )

    def forward(self, x):
        return self.head(x)


# ----------------------------------------------------------------------------
# Swin Transformer Class


class SwinV2(AbstractNetwork):
    def __init__(
        self,
        img_resolution: Union[int, tuple[int, int], list[int], ListConfig],
        in_channels: int,
        out_channels: int,
        window_size: Union[int, tuple[int, int], list[int], ListConfig],
        shift_size: Union[int, tuple[int, int], list[int], ListConfig],
        patch_size: Union[int, tuple[int, int], list[int], ListConfig],
        depth: int = 6,
        dim: int = 512,
        heads: int = 12,
        auxiliary_dim: int = 0,
        flash: bool = True,
        logvar: bool = False,
        timestep_weight: float = 1.0,
    ):
        super().__init__(img_resolution, in_channels, out_channels)
        image_height, image_width = Shape2D(img_resolution).shape
        patch_height, patch_width = Shape2D(patch_size).shape
        grid_size = gh, gw = (image_height // patch_height, image_width // patch_width)
        self.auxiliary_dim = auxiliary_dim
        self.timestep_weight = timestep_weight

        self.pos_embed = nn.Parameter(torch.randn(1, gh * gw, dim) * 0.02)
        self.patch_embed = PatchEmbedding(in_channels, patch_size, dim)
        self.latent_embed = LatentEmbedding(dim)
        self.logvar_embed = nn.Linear(dim, 1) if logvar else None
        self.auxiliary_embed = nn.Linear(auxiliary_dim, dim) if auxiliary_dim else None
        self.transformer = SwinTransformer(
            depth,
            dim,
            heads,
            window_size,
            grid_size,
            shift_size,
            flash=flash,
        )
        self.head = OutputHead(dim, out_channels, patch_size, grid_size)
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "modulation" in name or "head" in name:  # start with layer norm
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        auxiliary: Optional[torch.Tensor] = None,
        jvp: bool = False,
        return_logvar: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.patch_embed(x)  # b, n, d
        x = x + self.pos_embed  # new: nersc swinv2

        if t.dim() == 0 or (t.dim() == 1 and t.size(0) == 1):
            t = t.repeat(x.size(0))
        t = timestep_embedding(t * self.timestep_weight, x.size(2))  # b, d
        if self.auxiliary_embed is not None and auxiliary is not None:
            t = t + self.auxiliary_embed(auxiliary * math.sqrt(self.auxiliary_dim))
        t = self.latent_embed(t)

        x = self.transformer(x, t, jvp)  # b, n, d
        x = self.head(x)  # b, c, h, w

        if self.logvar_embed and return_logvar:
            logvar = self.logvar_embed(t).squeeze(-1)
            return x, logvar

        return x


if __name__ == "__main__":
    import ezpz

    device = ezpz.get_torch_device()
    x = torch.randn(1, 69 * 2, 128, 256).to(device)
    t = torch.randn(x.shape[0]).to(device)

    model = SwinV2(
        img_resolution=x.shape[2:],
        in_channels=x.shape[1],
        out_channels=x.shape[1],
        window_size=[16, 16],
        shift_size=[8, 8],
        patch_size=[1, 1],
        depth=12,
        dim=1056,
        heads=12,
        logvar=True,
    ).to(device)

    print(f"=> {sum(p.numel() for p in model.parameters() if p.requires_grad)} params")

    y = model(x, t)
    assert y.shape == x.shape

    y, logvar = model(x, t, return_logvar=True)
    print(y.shape, logvar.shape, t.shape)
