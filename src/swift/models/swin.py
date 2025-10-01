from typing import Optional, Union

import ezpz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules
from omegaconf import ListConfig

if torch.xpu.is_available():
    import intel_extension_for_pytorch
    import oneccl_bindings_for_pytorch

import math

from einops import rearrange
from einops.layers.torch import Rearrange

from swift.models.abstract import AbstractNetwork, Shape2D

logger = ezpz.get_logger(__name__)

# ----------------------------------------------------------------------------
# Utility Functions


def get_swin_flop_count(
    img_shape: Union[tuple[int, int], list[int]],
    batch_size: int,
    depth: int,
    num_channels: int,
    hidden_size: int,
    ffn_hidden_size: int,
    patch_size: Union[tuple[int, int], list[int]],
    window_size: Union[tuple[int, int], list[int]],
) -> int:
    """Compute the flop counts of the model."""
    img_h, img_w = img_shape
    p_dim = patch_size[0] * patch_size[1]
    seqlen_per_window = window_size[0] * window_size[1]  # seq length per window
    assert img_h % (window_size[0] * patch_size[0]) == 0
    assert img_w % (window_size[1] * patch_size[1]) == 0
    nwindows_per_batch = img_h * img_w / seqlen_per_window / p_dim
    nwindows = batch_size * nwindows_per_batch  # total num windows

    pre_and_post_process = 2 * nwindows * p_dim * num_channels * hidden_size
    qkvo = 4 * nwindows * seqlen_per_window * hidden_size**2
    fa = 2 * nwindows * seqlen_per_window**2 * hidden_size
    glu = (  # fused (Gate + Up Proj) + Down Proj
        3 * nwindows * seqlen_per_window * ffn_hidden_size * hidden_size
    )
    fwd_flop = (qkvo + fa + glu) * depth + pre_and_post_process

    return int(6 * fwd_flop)  # 3x for fwd+bwd, 2x for mult+add operations


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
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.get_default_dtype())
        / half
    ).to(device=t.device)
    args = t[:, None].to(torch.get_default_dtype()) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    embedding = (
        embedding.reshape(embedding.shape[0], 2, -1).flip(1).reshape(*embedding.shape)
    )  # flip sin/cos as done with edm

    return embedding


# ----------------------------------------------------------------------------
# Helper Classes
#
class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


class LatentEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l1 = nn.Linear(dim, dim, bias=True)
        self.l2 = nn.Linear(dim, dim, bias=True)

    def forward(self, emb):
        return F.silu(self.l2(F.silu(self.l1(emb))))


class ModulatedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # self.norm = torch.nn.modules.normalization.RMSNorm(dim)
        self.norm = RMSNorm(dim, eps)
        self.modulation = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x, t):
        x = self.norm(x)  # b, n, d
        scale, shift = self.modulation(t).chunk(2, dim=-1)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


class PositionalEncoding2D(nn.Module):
    """https://github.com/tatp22/multidim-positional-encoding"""

    def __init__(self, channels, max_positions=10_000):
        super().__init__()
        self.channels = int(math.ceil(channels / 4) * 2)
        inv_freq = 1.0 / (
            max_positions ** (torch.arange(0, self.channels, 2).float() / self.channels)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def _get_emb(self, sin_inp):
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, x):
        assert x.ndim == 4, "input has to be 4d!"
        if self.cached_penc is not None and self.cached_penc.shape == x.shape:
            return self.cached_penc

        b, c, h, w = x.shape
        pos_x = torch.arange(h, device=x.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(w, device=x.device, dtype=self.inv_freq.dtype)
        sin_inp_x = pos_x.unsqueeze(1) * self.inv_freq
        sin_inp_y = pos_y.unsqueeze(1) * self.inv_freq
        emb_x = self._get_emb(sin_inp_x)
        emb_y = self._get_emb(sin_inp_y)

        emb_x = emb_x.unsqueeze(1).expand(h, w, self.channels)
        emb_y = emb_y.unsqueeze(0).expand(h, w, self.channels)

        emb = torch.cat([emb_x, emb_y], dim=-1)
        emb = emb[..., :c].permute(2, 0, 1)
        self.cached_penc = emb.unsqueeze(0).repeat(b, 1, 1, 1).to(x.dtype)
        return self.cached_penc


class RoPE2D(nn.Module):
    """Axial Frequency 2D Rotary Positional Embeddings (https://arxiv.org/pdf/2403.13298).

    The embedding is applied to the x-axis and y-axis separately.
    """

    def __init__(
        self,
        window_size: tuple[int, int],
        rope_dim: int,
        rope_base: int = 10_000,
    ):
        super().__init__()
        self.window_size = window_size
        self.rope_dim = rope_dim
        self.rope_base = rope_base
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.rope_base
            ** (
                torch.arange(0, self.rope_dim, 2)[: (self.rope_dim // 2)].float()
                / self.rope_dim
            )
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache()

    def build_rope_cache(self):
        wh, ww = self.window_size
        patches_per_tile = wh * ww

        patch_idx = torch.arange(
            patches_per_tile, dtype=self.theta.dtype, device=self.theta.device
        )
        patch_x_pos = patch_idx % ww
        patch_y_pos = patch_idx // ww

        x_theta = torch.einsum("i, j -> ij", patch_x_pos, self.theta).float()
        y_theta = torch.einsum("i, j -> ij", patch_y_pos, self.theta).float()

        freqs = torch.cat([x_theta, y_theta], dim=-1)
        cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xdtype = x.dtype  # b, h, n, d

        x = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = self.cache[None, None, :, :, :]

        x = torch.stack(
            [
                x[..., 0] * rope_cache[..., 0] - x[..., 1] * rope_cache[..., 1],
                x[..., 1] * rope_cache[..., 0] + x[..., 0] * rope_cache[..., 1],
            ],
            dim=-1,
        )
        x = x.flatten(3)
        return x.to(xdtype)


class FeedForward(nn.Module):
    """SwiGLU FeedForward"""

    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.norm = ModulatedRMSNorm(dim)
        self.w1 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, t):
        x = self.norm(x, t)
        gate, up_proj = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up_proj)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, **rope_kwargs):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim**-0.5

        self.norm = ModulatedRMSNorm(dim)
        self.rope = RoPE2D(**rope_kwargs)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.wo = nn.Linear(inner_dim, dim, bias=False)
        self.attn_fn = self.optimized_attention

    def naive_attention(self, q, k, v):
        attn = (q * self.scale) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out

    def optimized_attention(self, q, k, v):
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=self.scale
        )

    def forward(self, x, t):
        x = self.norm(x, t)
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b n (h d) -> b h n d", h=self.heads)

        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.rope(q), self.rope(k)

        out = self.attn_fn(q, k, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.wo(out)


class SwinTransformer(nn.Module):
    def __init__(
        self,
        depth,
        dim,
        heads,
        head_dim,
        mlp_dim,
        patch_size,
        window_size,
        grid_size,
        shift_size,
        rope_base,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.window_size = window_size
        self.grid_size = grid_size
        self.shift_size = shift_size

        rope_kwargs = {
            "window_size": self.window_size,
            "rope_dim": head_dim // 2,
            "rope_base": rope_base,
        }

        self.layers = nn.Sequential(
            *[
                nn.ModuleList(
                    [
                        Attention(dim, heads, head_dim, **rope_kwargs),
                        FeedForward(dim, mlp_dim),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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

            x = attn(x, t_expanded)  # num_windows * b, n, d

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
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class OutputHead(nn.Module):
    def __init__(self, dim, out_channels, patch_size, grid_size):
        super().__init__()
        p1, p2 = patch_size
        gh, gw = grid_size

        self.norm = ModulatedRMSNorm(dim)  # b, n, d
        self.head = nn.Sequential(
            nn.Linear(dim, out_channels * p1 * p2, bias=False),  # b, n, c*p1*p2
            Rearrange(
                "b (h w) (c p1 p2) -> b c (h p1) (w p2)", p1=p1, p2=p2, h=gh, w=gw
            ),
        )

    def forward(self, x, t):
        x = self.norm(x, t)
        out = self.head(x)
        return out


# ----------------------------------------------------------------------------
# Swin Transformer Class
#


class Swin(AbstractNetwork):
    def __init__(
        self,
        img_resolution: Union[int, tuple[int, int], list[int], ListConfig],
        in_channels: int,
        out_channels: int,
        window_size: Union[  # number of patches in a window
            int, tuple[int, int], list[int], ListConfig
        ],
        shift_size: Union[  # patches to shift
            int, tuple[int, int], list[int], ListConfig
        ],
        patch_size: Union[int, tuple[int, int], list[int], ListConfig],
        depth: int = 6,
        dim: int = 512,
        heads: int = 12,
        head_dim: int = 64,
        mlp_dim: int = 512,
        rope_base: int = 10_000,
        auxiliary_dim: int = 0,  # dimension of auxiliary input, 0 = none
    ):
        super().__init__(img_resolution, in_channels, out_channels)
        patch_size = (16, 16) if patch_size is None else patch_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = Shape2D(window_size).shape
        self.window_height, self.window_width = self.window_size
        self.shift_size = Shape2D(shift_size).shape
        self.patch_size = Shape2D(patch_size).shape
        self.patch_height, self.patch_width = self.patch_size
        self.depth = depth
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.mlp_dim = mlp_dim
        self.rope_base = rope_base
        self.auxiliary_dim = auxiliary_dim
        self.img_resolution = Shape2D(img_resolution).shape
        self.image_height, self.image_width = self.img_resolution
        self.grid_size = (
            self.image_height // self.patch_height,
            self.image_width // self.patch_width,
        )

        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_windows = self.num_patches // (self.window_height * self.window_width)
        self.ape = PositionalEncoding2D(in_channels)
        self.patch_embed = PatchEmbedding(in_channels, self.patch_size, dim)
        self.latent_embed = LatentEmbedding(dim)
        self.auxiliary_embed = nn.Linear(auxiliary_dim, dim) if auxiliary_dim else None
        self.auxiliary_dim = auxiliary_dim

        self.transformer = SwinTransformer(
            depth,
            dim,
            heads,
            head_dim,
            mlp_dim,
            self.patch_size,
            self.window_size,
            self.grid_size,
            self.shift_size,
            rope_base,
        )
        self.single_sample_flop = self.get_flop_count(1)
        self.head = OutputHead(dim, out_channels, self.patch_size, self.grid_size)
        self._init_swin_weights(dim, depth)

    def _init_swin_weights(self, dim: int, depth: int):
        """(Loss) Spike No More: https://arxiv.org/abs/2312.16903"""
        # section 4
        sigma = math.sqrt(2.0 / (5 * dim))
        scale = math.sqrt(1.0 / (2 * depth))

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "wo" in name or "w2" in name:  # attn/fcn output projs, Eq 15 and 20
                    nn.init.normal_(module.weight, mean=0, std=sigma * scale)
                else:  # standard init for other linear layers
                    nn.init.normal_(module.weight, mean=0, std=sigma)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):  # patch embedding
                nn.init.normal_(module.weight, mean=0, std=sigma)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_flop_count(self, batch_size: int) -> int:
        """Compute the flop counts of the model."""
        return get_swin_flop_count(
            batch_size=batch_size,
            depth=self.depth,
            hidden_size=self.dim,
            num_channels=self.in_channels,
            ffn_hidden_size=self.head_dim,
            patch_size=list(self.patch_size),
            window_size=list(self.window_size),
            img_shape=list(self.img_resolution),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, auxiliary: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.ape(x)  # b, c, h, w
        x = self.patch_embed(x)  # b, n, d

        if t.dim() == 0 or (t.dim() == 1 and t.size(0) == 1):
            t = t.repeat(x.size(0))
        t = timestep_embedding(t, x.size(2))  # b, d
        if self.auxiliary_embed is not None and auxiliary is not None:
            t = t + self.auxiliary_embed(auxiliary * math.sqrt(self.auxiliary_dim))
        t = self.latent_embed(t)

        x = self.transformer(x, t)  # b, n, d
        x = self.head(x, t)  # b, c, h, w
        return x


if __name__ == "__main__":
    if torch.xpu.is_available():
        torch.set_default_device("xpu")
    elif torch.cuda.is_available():
        torch.set_default_device("cuda")
    else:
        raise KeyboardInterrupt("No device available")
    torch.set_default_dtype(torch.bfloat16)
    B, c, h, w = 1, 73, 1536, 768  # img dim
    hc = 12  # head count
    nlayers = 4  # num layers
    d = 1024  # hidden dim
    d_ffnn = 4 * d  # ffnn hidden dim
    patch_size = [32, 8]
    window_size = [16, 16]
    x = torch.randn(B, c, h, w)
    t = torch.randn(B)

    # model_flop_count = (
    #     get_swin_flop_count(
    #         img_dim=[h, w],
    #         B=B,
    #         num_layers=nlayers,
    #         c=c,
    #         d=d,
    #         d_ffnn=d_ffnn,
    #         p=patch_size,
    #         window_size=window_size,
    #     )
    #     / 3
    # )  # divide by three for fwd only flop
    model = Swin(
        img_resolution=[h, w],
        in_channels=c,
        out_channels=1,
        window_size=window_size,
        shift_size=(1, 2),  #
        patch_size=patch_size,
        depth=nlayers,
        dim=d,
        heads=hc,
        head_dim=d,
        mlp_dim=d_ffnn,
    )
    if ezpz.get_rank() == 0:
        from torchinfo import summary

        summary(model, depth=3)  # , input_size=x.shape)

    model_flop_count = model.get_flop_count(batch_size=B)
    _ = model(x, t)  # warmup
    # compiled_model = model.compile()

    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"=> Trainable Params: {nparam}")
    logger.info(
        f"=> sequence len per window: {window_size[0] * window_size[1] * patch_size[0] * patch_size[1]}"
    )
    import time

    dkernel = None
    peak_memory = None
    if torch.cuda.is_available():
        dkernel = torch.cuda
    elif torch.xpu.is_available():
        dkernel = torch.xpu

    if dkernel is not None:
        dkernel.synchronize()
    strt = time.perf_counter()
    out = model(x, t)
    if dkernel is not None:
        dkernel.synchronize()
    end = time.perf_counter()
    tflops = model_flop_count / 1e12 / (end - strt)
    if dkernel is not None:
        try:
            peak_memory = dkernel.max_memory_reserved(0) / 1024**3  # type:ignore
        except Exception as exc:
            logger.exception(f"Failed to get peak memory\n{exc}")

    logger.info(f"=> tflops: {tflops}")
    logger.info(f"=> model_tflop_count: {model_flop_count / 1e12}")
    if peak_memory is not None:
        logger.info(f"=> peak_memory: {peak_memory}")
