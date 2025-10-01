import math
from contextlib import contextmanager
from functools import partial
from typing import Tuple

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

PRESSURE_LEVEL_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "wind_speed",
    "temperature",
    "relative_humidity",
    "specific_humidity",
    "vorticity",
    "potential_vorticity",
]

# fmt: off
DEFAULT_PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
# fmt: on


def _calculate_latitude_weights(lat_dim: int) -> torch.Tensor:
    w_lat = torch.cos(torch.deg2rad(torch.linspace(-90, 90, lat_dim)))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.clamp(w_lat, min=0.1)
    return w_lat.view(1, 1, -1, 1)


def _calculate_variable_weights(variables: list[str]) -> torch.Tensor:
    single_level_weight_dict = {
        "2m_temperature": 1.0,
        "sea_surface_temperature": 0.1,
        "10m_u_component_of_wind": 0.1,
        "10m_v_component_of_wind": 0.1,
        "mean_sea_level_pressure": 0.1,
    }

    pressure_weights = [
        l / sum(DEFAULT_PRESSURE_LEVELS) for l in DEFAULT_PRESSURE_LEVELS
    ]
    pressure_level_weight_dict = {}
    for var in PRESSURE_LEVEL_VARS:
        for l, w in zip(DEFAULT_PRESSURE_LEVELS, pressure_weights):
            pressure_level_weight_dict[var + "_" + str(l)] = w

    weights = {**single_level_weight_dict, **pressure_level_weight_dict}
    weights = torch.Tensor([weights[var] for var in variables]).view(1, -1, 1, 1)
    weights = weights / weights.sum()
    return weights


# ----------------------------------------------------------------------------


def lognormal(x: torch.Tensor, P_mean: float, P_std: float) -> torch.Tensor:
    n = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
    return torch.exp((n * P_std + P_mean))


def loguniform(x: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    sigma_min = torch.tensor(sigma_min, device=x.device)
    sigma_max = torch.tensor(sigma_max, device=x.device)
    u = torch.rand([x.shape[0], 1, 1, 1], device=x.device)
    us = torch.log(sigma_min) + u * (torch.log(sigma_max) - torch.log(sigma_min))
    return torch.exp(us)


NOISE_SAMPLING_METHODS = {
    "lognormal": lognormal,
    "loguniform": loguniform,
}

# ----------------------------------------------------------------------------


@contextmanager
def disable_forward_hooks(module):
    saved = module._forward_hooks
    module._forward_hooks = {}
    try:
        yield
    finally:
        module._forward_hooks = saved


# ----------------------------------------------------------------------------


class EDMLoss(torch.nn.Module):
    """Elucidating Diffusion Models (EDM) Loss"""

    def __init__(self, dataset, noise: dict[str, float | str], sigma_data: float):
        super().__init__()
        self.cfg = noise.copy()
        self._sampling_fn = partial(
            NOISE_SAMPLING_METHODS[self.cfg.pop("dist")], **self.cfg
        )
        self.sigma_data = sigma_data

        self.register_buffer("w_lat", _calculate_latitude_weights(dataset._shape[1]))
        self.register_buffer("w_var", _calculate_variable_weights(dataset.variables))

    def forward(self, net, x, condition=None, auxiliary=None):
        sigma = self._sampling_fn(x)
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x) * sigma
        D_yn = net(x + n, sigma, condition, auxiliary)
        return (weight * (self.w_var * self.w_lat * (D_yn - x) ** 2)).sum(dim=1).mean()


class TrigFlowLoss(torch.nn.Module):
    """TrigFlow Diffusion Loss"""

    def __init__(self, dataset, noise: dict[str, float | str], sigma_data: float):
        super().__init__()
        self.cfg = noise.copy()
        self._sampling_fn = partial(
            NOISE_SAMPLING_METHODS[self.cfg.pop("dist")], **self.cfg
        )
        self.sigma_data = sigma_data

        self.register_buffer("w_lat", _calculate_latitude_weights(dataset._shape[1]))
        self.register_buffer("w_var", _calculate_variable_weights(dataset.variables))

    def forward(self, net, x, condition=None, auxiliary=None, **kwargs):
        tau = self._sampling_fn(x)
        t = torch.atan(tau / self.sigma_data)

        z = torch.randn_like(x) * self.sigma_data
        cos_t, sin_t = torch.cos(t), torch.sin(t)
        x_t = cos_t * x + sin_t * z
        v_t = cos_t * z - sin_t * x

        net_kwargs = dict(  # ddp things
            return_logvar=bool(
                getattr(getattr(net, "module", net).model, "logvar_embed", None)
            )
        )
        out = net(x_t / self.sigma_data, t, condition, auxiliary, **net_kwargs)
        if isinstance(out, tuple):
            F_x, logvar = out
            logvar = logvar.reshape(-1, 1, 1, 1)
        else:
            F_x, logvar = out, torch.zeros_like(x[:, 0:1, 0:1, 0:1])

        return (
            (
                (1 / torch.exp(logvar))
                * (self.w_var * self.w_lat * torch.square(self.sigma_data * F_x - v_t))
                + logvar
            )
            .sum(dim=1)
            .mean()
        )


class SCMLoss(torch.nn.Module):
    """Simplified and Stabilized Continuous-time Consistency Models (sCM) Loss"""

    def __init__(
        self,
        dataset,
        noise: dict[str, float | str],
        sigma_data: float,
        tangent_warmup_kimg: int = 0,
        distillation: bool = False,
    ):
        super().__init__()
        self.cfg = noise.copy()
        self._sampling_fn = partial(
            NOISE_SAMPLING_METHODS[self.cfg.pop("dist")], **self.cfg
        )
        self.sigma_data = sigma_data
        self.tangent_warmup_kimg = tangent_warmup_kimg
        self.distillation = distillation

        self.register_buffer("w_lat", _calculate_latitude_weights(dataset._shape[1]))
        self.register_buffer("w_var", _calculate_variable_weights(dataset.variables))

    def forward(
        self,
        net,
        x,
        step,
        condition=None,
        auxiliary=None,
        net_pretrained=None,
        **kwargs,
    ):
        tau = self._sampling_fn(x)
        t = torch.atan(tau / self.sigma_data)

        z = torch.randn_like(x) * self.sigma_data
        cos_t = torch.cos(t)
        sin_t = torch.sin(t)
        x_t = cos_t * x + sin_t * z

        if self.distillation and net_pretrained is not None:
            with torch.no_grad():  # assumes v-prediction
                dxt_dt = self.sigma_data * net_pretrained(
                    x_t / self.sigma_data, t, condition, auxiliary
                )
        else:
            dxt_dt = cos_t * z - sin_t * x

        def wrapper(x, t) -> Tuple[torch.Tensor, torch.Tensor]:
            return net.module(x, t, condition, auxiliary, jvp=True)  # unwrap module

        v_x = cos_t * sin_t * dxt_dt / self.sigma_data
        v_t = cos_t * sin_t
        with disable_forward_hooks(net.module):  # needed for wandb hooks
            _, dF_x = torch.func.jvp(
                wrapper, (x_t / self.sigma_data, t), (v_x, v_t), has_aux=False
            )
        net_kwargs = dict(  # ddp things
            return_logvar=bool(
                getattr(getattr(net, "module", net).model, "logvar_embed", None)
            )
        )
        out = net(x_t / self.sigma_data, t, condition, auxiliary, **net_kwargs)
        if isinstance(out, tuple):
            F_x, logvar = out
            logvar = logvar.reshape(-1, 1, 1, 1)
        else:
            F_x, logvar = out, torch.zeros_like(x[:, 0:1, 0:1, 0:1])
        r = (  # tangent warmup
            min(1.0, step / (self.tangent_warmup_kimg * 1000))
            if self.tangent_warmup_kimg > 0
            else 1.0
        )

        # JVP rearrangement
        # NOTE: 1 / (sigma_data * tan(t)) is encoded as the extra cos(t) term
        g = -(cos_t**2) * (self.sigma_data * F_x.detach() - dxt_dt) - r * (
            (cos_t * sin_t) * x_t + self.sigma_data * dF_x.detach()
        )

        # tangent normalization
        gn = torch.linalg.vector_norm(g, dim=(1, 2, 3), keepdim=True)
        gn = gn * np.sqrt(gn.numel() / g.numel())  # norm invariance to spatial dims
        g = g / (gn + 0.1)

        # sigma = torch.tan(t) * self.sigma_data
        weight = 1  # / sigma

        return (
            (
                (weight / torch.exp(logvar))
                * (self.w_var * self.w_lat * torch.square(F_x - F_x.detach() - g))
                + logvar
            )
            .sum(dim=1)
            .mean()
        )


# ----------------------------------------------------------------------------


class MSELoss(torch.nn.Module):
    """Multistep MSE Loss"""

    def __init__(self, dataset, sigma_data: float):
        super().__init__()
        self.dataset = dataset
        self.sigma_data = sigma_data

        self.register_buffer("w_lat", _calculate_latitude_weights(dataset._shape[1]))
        self.register_buffer("w_var", _calculate_variable_weights(dataset.variables))

    def forward(
        self,
        net: torch.nn.Module,
        target: torch.Tensor,
        condition: torch.Tensor,
        auxiliary: torch.Tensor | None = None,
        steps: int = 1,
        idx: list[int] = None,
    ):
        # tau = torch.exp(torch.log(200, device=target.device))
        # t = torch.atan(tau / self.sigma_data)
        t = torch.tensor(torch.pi / 2, device=target.device, dtype=target.dtype)

        cond = condition
        for _ in range(steps):
            x_t = torch.randn_like(target) * self.sigma_data

            out = net(x_t / self.sigma_data, t.expand(target.shape[0]), cond, auxiliary)
            # pred = torch.cos(t) * x_t - torch.sin(t) * self.sigma_data * out
            pred = self.sigma_data * out

            # NOTE: assumes residual training
            y_unstd = self.dataset.unstandardize_t(pred)
            x_unstd = self.dataset.unstandardize_x(cond)
            cond = self.dataset.standardize_x(x_unstd + y_unstd)

        return (self.w_var * self.w_lat * (pred - target) ** 2).sum(dim=1).mean()


class CRPSLoss(torch.nn.Module):
    """Multistep CRPS Loss

    adapted from: https://github.com/NCAR/miles-credit/blob/main/credit/losses/almost_fair_crps.py
    """

    def __init__(
        self,
        dataset,
        sigma_data: float,
        ensemble_size: int = 2,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.dataset = dataset
        self.sigma_data = sigma_data
        self.ensemble_size = ensemble_size
        self.alpha = alpha

        self.register_buffer("w_lat", _calculate_latitude_weights(dataset._shape[1]))
        self.register_buffer("w_var", _calculate_variable_weights(dataset.variables))

        self.batched_forward = torch.vmap(self._single_forward)

    def _single_forward(self, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target: shape (1, c, t, lat, lon)
            pred: shape (ensemble, c, t, lat, lon)
        Returns:
            crps: shape (c, t, lat, lon)
        """
        pred = torch.movedim(pred, 0, -1)  # (c, t, lat, lon, ensemble)
        target = target.squeeze(0)  # (c, t, lat, lon)

        return self._kernel_crps(pred, target, self.alpha)

    def _kernel_crps(self, preds: torch.Tensor, targets: torch.Tensor, alpha: float):
        """
        Args:
            preds: (c, t, lat, lon, ensemble)
            targets: (c, t, lat, lon)
        Returns:
            crps: (c, t, lat, lon)
        """
        m = preds.shape[-1]
        assert m > 1, "Ensemble size must be greater than 1."

        epsilon = (1.0 - alpha) / m

        # |X_i - y|
        skill = torch.abs(preds - targets.unsqueeze(-1)).mean(-1)

        # |X_i - X_j|
        pred1 = preds.unsqueeze(-2)  # (c, t, lat, lon, 1, m)
        pred2 = preds.unsqueeze(-1)  # (c, t, lat, lon, m, 1)
        pairwise_diffs = torch.abs(pred1 - pred2)  # (c, t, lat, lon, m, m)

        # Create diagonal mask to exclude i == j
        eye = torch.eye(m, dtype=torch.bool, device=preds.device)
        mask = ~eye.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,1,m,m)
        pairwise_diffs = pairwise_diffs * mask

        spread = (1.0 / (2 * m * (m - 1))) * torch.sum(pairwise_diffs, dim=(-1, -2))

        return skill - (1 - epsilon) * spread

    def _one_step(self, net, target, cond, auxiliary, idx, i):
        t = torch.tensor(torch.pi / 2, device=target.device, dtype=target.dtype)
        B = target.shape[0]
        delta = int(auxiliary[0] * 10)  # NOTE: assumes same delta

        x_t = torch.randn_like(target) * self.sigma_data

        # add forcings in to condition
        cond = torch.cat(
            [
                cond,
                self.dataset.standardize_x(
                    torch.stack(
                        [
                            self.dataset.get_forcings(j + int(i * dt * 10 // 6))
                            for j, dt in zip(idx, auxiliary)
                        ],
                        dim=0,
                    ).to(target.device)
                ),
            ],
            dim=1,
        )

        out = net(x_t / self.sigma_data, t.expand(B), cond, auxiliary)  # b, c, lat, lon
        # NOTE: only works with TrigFlow / SCM (v-prediction)
        pred = -self.sigma_data * out

        # residual update to condition
        y_unstd = self.dataset.unstandardize_t(pred, delta)
        x_unstd = self.dataset.unstandardize_x(cond, delta)[
            :, : len(self.dataset.variables)
        ]
        cond = self.dataset.standardize_x(x_unstd + y_unstd, delta)

        return cond, pred

    def forward(
        self,
        net: torch.nn.Module,
        target: torch.Tensor,
        condition: torch.Tensor,
        auxiliary: torch.Tensor,
        idx: list[int],
        steps: int = 1,
        chunk_size: int = 2,  # NOTE: adjust based on memory needs
    ):
        step_fns = [
            lambda c, i=i: self._one_step(net, target, c, auxiliary, idx, i)[0]
            for i in range(steps - 1)
        ]

        preds = []
        for _ in range(self.ensemble_size):
            cond = condition[:, : len(self.dataset.variables)]
            if steps > 1:
                cond = checkpoint_sequential(
                    step_fns,
                    math.ceil((steps - 1) / chunk_size),
                    cond,
                    use_reentrant=True,
                )
            _, pred = self._one_step(net, target, cond, auxiliary, idx, steps - 1)
            preds.append(pred)

        # NOTE: only a single timestep with t (dim 3)
        preds = torch.stack(preds, dim=1).unsqueeze(3)  # (b, ensemble, c, 1, lat, lon)
        target = target.unsqueeze(1).unsqueeze(3)  # (b, 1, c, 1, lat, lon)

        # NOTE: remove time dimension (dim 2)
        crps = self.batched_forward(target, preds).squeeze(2)  # (b, c, lat, lon)

        return (self.w_var * self.w_lat * crps).sum(dim=1).mean()
