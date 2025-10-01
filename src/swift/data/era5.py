import os
from glob import glob
from typing import Dict, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ERA5Dataset(Dataset):
    def __init__(
        self,
        root: str,
        variables: list[str],
        forcings: list[str] = [],
        intervals: list[int] = [6, 12, 24],
        split: str = "train",
        residual: bool = False,
    ):
        super().__init__()
        assert sorted(intervals) in (
            [6],
            [12],
            [24],
            [6, 12],
            [6, 24],
            [12, 24],
            [6, 12, 24],
        ), "must be combination of [6, 12, 24]"

        self.root = root
        self.files = sorted(glob(os.path.join(root, split, "*.h5")))
        self.variables = variables
        self.forcings = forcings

        self.intervals = intervals
        self.residual = residual
        self.x_means, self.x_stds, self.t_means, self.t_stds = self._setup_standardize()
        self._shape = self._load_file(
            self.files[np.random.randint(0, len(self.files))], variables
        ).shape

    @property
    def n_target_channels(self) -> int:
        assert len(self._shape) == 3
        return self._shape[0]

    @property
    def n_condition_channels(self) -> int:
        assert len(self._shape) == 3
        return self.n_target_channels + len(self.forcings)

    @property
    def img_resolution(self) -> tuple[int, int]:
        return self._shape[1], self._shape[2]

    def _load_file(self, path: str, variables: list[str]) -> np.ndarray:
        def _fill_nan(value: np.ndarray) -> np.ndarray:
            if np.isnan(value).any():
                np.copyto(value, np.nanmin(value), where=np.isnan(value))
            return value

        with h5py.File(path, "r") as f:
            data = {
                main_key: {
                    sub_key: _fill_nan(value[()])
                    for sub_key, value in group.items()
                    if sub_key in variables  # + ["time"]
                }
                for main_key, group in f.items()
                if main_key in ["input"]
            }
            return np.stack([data["input"][v] for v in variables], axis=0)

    def _load_and_stack(self, filename: str, variables: list[str]) -> np.ndarray:
        with np.load(os.path.join(self.root, filename)) as data:
            return np.stack([data[v] for v in variables], axis=0).reshape(-1, 1, 1)

    def _setup_standardize(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Union[Dict[int, np.ndarray], np.ndarray],
        Union[Dict[int, np.ndarray], np.ndarray],
    ]:
        x_means = self._load_and_stack(
            "normalize_mean.npz", self.variables + self.forcings
        )
        x_stds = self._load_and_stack(
            "normalize_std.npz", self.variables + self.forcings
        )

        if self.residual:
            t_stds = {
                i: self._load_and_stack(f"normalize_diff_std_{i}.npz", self.variables)
                for i in self.intervals
            }
            t_means = {i: np.zeros_like(t_stds[i]) for i in self.intervals}
        else:
            if len(self.intervals) > 1 and self.intervals[0] != 6:
                raise ValueError(
                    "Only 6h intervals are supported for standardization at the moment."
                )
            t_means, t_stds = x_means, x_stds

        return x_means, x_stds, t_means, t_stds

    def _transform_standardize(
        self,
        v: Union[np.ndarray, torch.Tensor],
        means: np.ndarray,
        stds: np.ndarray,
        inverse: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(v, torch.Tensor):
            m = torch.as_tensor(means, device=v.device, dtype=v.dtype)
            s = torch.as_tensor(stds, device=v.device, dtype=v.dtype)
        else:
            m, s = means, stds

        # NOTE: pseudo-dynamic transform of variables and forcings
        channels: int = v.shape[1 if v.ndim == 4 else 0]
        if channels == len(self.variables):
            m, s = m[: len(self.variables)], s[: len(self.variables)]
        elif channels == len(self.forcings):
            m, s = m[len(self.variables) :], s[len(self.variables) :]

        if inverse:
            return v * s + m
        else:
            return (v - m) / s

    def zero_field(self, x, delta: int = 6):
        channels: int = x.shape[1 if x.ndim == 4 else 0]
        if (
            delta == 24
            or "sea_surface_temperature" not in self.variables
            or channels == len(self.forcings)
        ):
            return x
        idx = self.variables.index("sea_surface_temperature")
        if x.ndim == 4:  # [B, C, H, W]
            x[:, idx, ...] = 0
        elif x.ndim == 3:  # [C, H, W]
            x[idx, ...] = 0
        return x

    def standardize_x(self, x, delta: int = 6):
        x = self._transform_standardize(x, self.x_means, self.x_stds)
        x = self.zero_field(x, delta)
        return x

    def unstandardize_x(self, x, delta: int = 6):
        x = self._transform_standardize(x, self.x_means, self.x_stds, inverse=True)
        x = self.zero_field(x, delta)
        return x

    def standardize_t(self, t, delta: int = 6):
        t = self._transform_standardize(t, self.t_means[delta], self.t_stds[delta])
        t = self.zero_field(t, delta)
        return t

    def unstandardize_t(self, t, delta: int = 6):
        t = self._transform_standardize(
            t, self.t_means[delta], self.t_stds[delta], inverse=True
        )
        t = self.zero_field(t, delta)
        return t

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.load(os.path.join(self.root, "lat.npy")).astype(np.float32)
        lon = np.load(os.path.join(self.root, "lon.npy")).astype(np.float32)
        return lat, lon

    def get_time(self, idx: int) -> np.datetime64:
        with h5py.File(self.files[idx], "r") as f:
            timestamp = f["input"]["time"][()]
            assert isinstance(timestamp, bytes)
            return np.datetime64(timestamp.decode("utf-8"))

    def get_forcings(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self._load_file(self.files[idx], self.forcings)).float()

    def __len__(self) -> int:
        # TODO: truncate by a factor of k finetuning steps
        return len(self.files[: -(max(self.intervals) * 1 // 6)])

    def __getitem__(
        self, spec: int | tuple[int, int] | tuple[int, int, int]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[int, torch.Tensor]]:
        if isinstance(spec, tuple):
            spec = tuple(int(i) for i in spec)
        else:
            spec = int(spec)

        match spec:
            case int() as idx:
                offset, delta = 1, None
            case (int() as idx, int() as off):
                offset, delta = off, None
            case (int() as idx, int() as off, int() as d):
                offset, delta = off, d
            case _:
                raise ValueError(f"Invalid index spec: {spec!r}")

        if delta is None:
            delta = np.random.choice(self.intervals)

        x = self._load_file(self.files[idx], self.variables + self.forcings)
        t = self._load_file(self.files[idx + (offset * delta // 6)], self.variables)

        if self.residual:
            x_prev = (
                self._load_file(
                    self.files[idx + (offset - 1) * delta // 6], self.variables
                )
                if offset > 1
                else x[: len(self.variables)]
            )
            t = t - x_prev

        x = torch.from_numpy(self.standardize_x(x, delta)).float()  # C+F x H x W
        t = torch.from_numpy(self.standardize_t(t, delta)).float()  # C x H x W

        return (x, t), (idx, torch.tensor(delta / 10.0).float())


class ERA5RollOutDataset(ERA5Dataset):
    def __init__(self, interval: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval = interval

    def __len__(self) -> int:
        return len(self.files[: -self.interval])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        original_idx = idx
        x = torch.from_numpy(
            self.standardize_x(self._load_file(self.files[idx], self.variables))
        ).float()

        # stack targets that represents days
        num_interval_per_day = 4
        assert self.interval >= num_interval_per_day, "cannot even predict one day"
        strt_idx = idx + num_interval_per_day
        t_lst = [
            self._load_file(self.files[idx + 1], self.variables)
        ]  # include 6h at start. TODO: fix this if 6h is not the base interval
        for i in range(strt_idx, strt_idx + self.interval, num_interval_per_day):
            t_lst.append(self._load_file(self.files[i], self.variables))
        t_stacked = np.stack(t_lst, axis=0)
        t = torch.from_numpy(t_stacked).float()  # unstandardized

        return x, t, original_idx  # C H W, interval/4+1 C H W, 1


def get_fallback_path():
    try:
        import ezpz

        machine = str(ezpz.get_machine()).lower()
    except Exception:
        logger.info(
            "[WARNING!] Unable to import ezpz! "
            "using fallback path for machine 'polaris'"
        )
        machine = "polaris"
    return {
        "aurora": "/flare/datasets/wb2/5.625deg_1_step_6hr_h5df/",
        "polaris": "/lus/eagle/projects/MDClimSim/jstock/data/wb2/5.625deg_1_step_6hr_h5df",
        "sophia": "/lus/eagle/projects/MDClimSim/jstock/data/wb2/5.625deg_1_step_6hr_h5df",
    }[machine]


if __name__ == "__main__":
    import logging
    import sys

    import matplotlib.pyplot as plt

    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    rootfp = sys.argv[1] if len(sys.argv) > 1 else get_fallback_path()
    dataset = ERA5Dataset(
        root=rootfp,
        split="train",
        residual=True,
        variables=[
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
    )
    m = len(dataset)
    logger.info(f"Loaded dataset with {m} samples from:\n  {rootfp}")
    (x, t), delta = dataset[m - 1]
    logger.info(f"{delta=}")
    logger.info(f"{x.shape}, {t.shape}")
    logger.info(f"{x.mean()=}, {x.std()=}")
    logger.info(f"{t.mean()=}, {t.std()=}")
    logger.info(
        f"{dataset.n_target_channels=}, {dataset.img_resolution=}, {len(dataset)=}"
    )

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(x[0], cmap="coolwarm")
    ax[1].imshow(t[0], cmap="coolwarm")
    ax[2].imshow(t[0] - x[0], cmap="coolwarm")
    for a in ax:
        a.axis("off")

    ax[0].set_title("x")
    ax[1].set_title("t")
    ax[2].set_title("t - x")

    fig.tight_layout()
    from pathlib import Path

    from swift import PROJECT_DIR

    resolution = "x".join([str(i) for i in dataset.img_resolution])
    outfile = Path(PROJECT_DIR).joinpath(
        f"media/era5-{resolution}-{dataset.n_target_channels}.png"
    )
    logger.info(f"Saving image to: {outfile}")
    # fig.savefig(outfile, bbox_inches="tight", dpi=300)
