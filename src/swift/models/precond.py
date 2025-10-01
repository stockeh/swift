import ezpz
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from swift.models.abstract import AbstractNetwork, _Shape2D

logger = ezpz.get_logger(__name__)


def _2d_resolution(x):
    if isinstance(x, int):
        return np.array([x, x], dtype=int)
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=int)
    assert x.shape[0] == 2
    return x


def _process_auxiliary(auxiliary, auxiliary_dim, batch_size, device):
    if auxiliary_dim == 0:
        return None
    elif auxiliary is None:
        return torch.zeros([1, auxiliary_dim], device=device)
    else:
        if not isinstance(auxiliary, torch.Tensor):
            auxiliary = torch.tensor(auxiliary, device=device)
        if auxiliary.dim() == 0 or (auxiliary.dim() == 1 and auxiliary.size(0) == 1):
            auxiliary = auxiliary.repeat(batch_size)
        return auxiliary.reshape(-1, auxiliary_dim)


# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        model_config: DictConfig,
        img_resolution: _Shape2D,
        img_channels: int,
        condition_channels: int = 0,  # Number of condition channels, 0 = unconditional.
        auxiliary_dim: int = 0,  # Number of class auxiliarys, 0 = none.
        sigma_min: float = 0.0,  # Minimum supported noise level.
        sigma_max: float = float("inf"),  # Maximum supported noise level.
        sigma_data: float = 0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.img_resolution = _2d_resolution(img_resolution)
        self.img_channels = img_channels
        self.condition_channels = condition_channels
        self.auxiliary_dim = auxiliary_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model_config = model_config
        # one of `swift.models.*` modules
        self.model: AbstractNetwork = instantiate(
            model_config,
            img_resolution=img_resolution,
            in_channels=img_channels + condition_channels,
            out_channels=img_channels,
            # other
            auxiliary_dim=auxiliary_dim,
            _convert_="object",
        )

    def forward(self, x, sigma, condition=None, auxiliary=None, **model_kwargs):
        # TODO: need we push x, sigma, auxiliary, F_x .to(torch.float32) ?
        # EDM sets these values manually, but we're using torch.autocast
        sigma = sigma.reshape(-1, 1, 1, 1)

        auxiliary = _process_auxiliary(
            auxiliary, self.auxiliary_dim, x.size(0), x.device
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        arg = c_in * x
        if condition is not None and self.condition_channels > 0:
            arg = torch.cat([arg, condition], dim=1)

        F_x = self.model(arg, c_noise.flatten(), auxiliary=auxiliary, **model_kwargs)
        D_x = c_skip * x + c_out * F_x
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Placeholder preconditioner for models that do not require preconditioning.


class PassPrecond(torch.nn.Module):
    def __init__(
        self,
        model_config: DictConfig,
        img_resolution: _Shape2D,
        img_channels: int,
        condition_channels: int = 0,  # Number of condition channels, 0 = unconditional.
        auxiliary_dim: int = 0,  # Number of class auxiliarys, 0 = none.
        sigma_min: float = 0.0,  # Minimum supported noise level.
        sigma_max: float = float("inf"),  # Maximum supported noise level.
        sigma_data: float = 1.0,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.img_resolution = _2d_resolution(img_resolution)
        self.img_channels = img_channels
        self.condition_channels = condition_channels
        self.auxiliary_dim = auxiliary_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model_config = model_config
        # one of `swift.models.*` modules
        self.model: AbstractNetwork = instantiate(
            model_config,
            img_resolution=img_resolution,
            in_channels=img_channels + condition_channels,
            out_channels=img_channels,
            # other
            auxiliary_dim=auxiliary_dim,
            _convert_="object",
        )

    def forward(self, x, t, condition=None, auxiliary=None, **model_kwargs):

        auxiliary = _process_auxiliary(
            auxiliary, self.auxiliary_dim, x.size(0), x.device
        )

        arg = x
        if condition is not None and self.condition_channels > 0:
            arg = torch.cat([arg, condition], dim=1)

        return self.model(
            arg,
            t.flatten(),
            auxiliary=auxiliary,
            **model_kwargs,
        )

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


if __name__ == "__main__":
    from torchinfo import summary

    from swift.models.swin import get_swin_flop_count

    torch.set_default_device(ezpz.get_torch_device_type())
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
    model_config = DictConfig(
        dict(
            _target_="swift.models.swin.Swin",
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
    )

    precond = EDMPrecond(
        model_config, img_resolution=[h, w], img_channels=c, condition_channels=c
    )

    if ezpz.get_rank() == 0:
        summary(precond, depth=3)

    model_flop_count = get_swin_flop_count(
        img_shape=(h, w),
        batch_size=B,
        depth=nlayers,
        num_channels=c,
        hidden_size=d,
        ffn_hidden_size=d_ffnn,
        patch_size=patch_size,
        window_size=window_size,
    )
    _ = precond(x, t)  # warmup
    # compiled_model = model.compile()

    nparam = sum(p.numel() for p in precond.parameters() if p.requires_grad)
    print(f"=> Trainable Params: {nparam}")
    print(
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
    out = precond(x, t)
    if dkernel is not None:
        dkernel.synchronize()
    end = time.perf_counter()
    tflops = model_flop_count / 1e12 / (end - strt)
    if dkernel is not None:
        try:
            peak_memory = dkernel.max_memory_reserved(0) / 1024**3  # type:ignore
        except Exception as exc:
            logger.exception(f"Failed to get peak memory\n{exc}")

    print(f"=> tflops: {tflops}", flush=True)
    print(f"=> model_tflop_count: {model_flop_count / 1e12}", flush=True)
    if peak_memory is not None:
        print(f"=> peak_memory: {peak_memory}", flush=True)
