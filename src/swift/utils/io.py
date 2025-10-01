from mpi4py import MPI  # isort:skip

import inspect
import logging
import os
import re
from collections import defaultdict

import dask.array as da
import ezpz
import numpy as np
import xarray as xr
import zarr

logging.basicConfig(level=logging.INFO)

_logger_cache = {}
_logger_modes = {"info", "debug", "warning", "error", "critical"}


COLOR_LOGS = os.environ.get("COLOR_LOGS", False) and not os.environ.get(
    "NO_COLOR", None
)


def log0(*args, mode="info", **kwargs):
    """
    Logs messages for rank 0 only, with the caller's context.

    Parameters:
    - args: Message arguments to log.
    - mode: Logging level/mode ('info', 'debug', 'warning', 'error', 'critical').

    Example:
        log0("This is a test message", mode="debug")
    """
    if ezpz.get_rank() == 0:
        if mode not in _logger_modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Supported modes are: {_logger_modes}"
            )

        parent = inspect.currentframe()
        frame = getattr(parent, "f_back", None)
        assert parent is not None and frame is not None
        try:
            code = frame.f_code
            key = (code.co_filename, code.co_name)

            if key not in _logger_cache:
                filename = os.path.splitext(os.path.basename(code.co_filename))[0]
                if COLOR_LOGS:
                    _logger_cache[key] = ezpz.get_logger(f"{filename}.{code.co_name}")
                else:
                    _logger_cache[key] = logging.getLogger(f"{filename}.{code.co_name}")

            logger = _logger_cache[key]
            log_function = getattr(logger, mode)
            log_function(" ".join(map(str, args)), **kwargs)

        finally:
            del frame


def print0(*args, **kwargs):
    if ezpz.get_rank() == 0:
        print(*args, **kwargs)


# ----------------------------------------------------------------------------


def compress_variables(variables):
    compressed = defaultdict(list)
    for var in variables:
        match = re.match(r"^(.*)_(\d+)$", var)
        if match:
            base_name, number = match.groups()
            compressed[base_name].append(int(number))
        else:
            compressed[var] = []
    return dict(compressed)


def create_empty_zarr(
    ofile: str,  # output path
    dataset,  # an object with methods get_lat_lon() and get_time()
    members: int,  # number of ensemble members
    steps: int,  # number of prediction lead-time steps
    indices: list | None = None,  # optional indices for a subset of samples
):
    """Create an empty Zarr dataset with the following structure:
    (time, number, prediction_timedelta, (level), latitude, longitude)

    read with `xr.open_zarr(ofile, decode_timedelta=True)`
    """
    n_samples = len(dataset)
    if indices is not None:
        assert len(indices) == n_samples
    else:
        indices = np.arange(n_samples, dtype=np.int32)

    lat, lon = dataset.get_lat_lon()
    time_coord = np.array(
        [dataset.get_time(i) for i in indices], dtype="datetime64[ns]"
    )
    pred_td = (np.arange(steps + 1) * np.timedelta64(6 * dataset.interval, "h")).astype(
        "timedelta64[ns]"
    )

    coords = {
        "time": (("time",), time_coord),
        "number": (("number",), np.arange(members, dtype=np.int32)),
        "prediction_timedelta": (("prediction_timedelta",), pred_td),
        "latitude": (("latitude",), lat),
        "longitude": (("longitude",), lon),
    }

    compressed_variables = compress_variables(dataset.variables)
    n_levels = max((len(levels) for levels in compressed_variables.values()), default=0)

    if n_levels:
        coords["level"] = (("level",), np.arange(n_levels, dtype=np.int32))

    base_dims = (
        "time",
        "number",
        "prediction_timedelta",
        "latitude",
        "longitude",
    )
    base_shape = (n_samples, members, steps + 1, len(lat), len(lon))
    # NOTE: (1, members, 1, len(lat), len(lon)) has possible issues
    base_chunks = (1, 1, 1, len(lat), len(lon))

    data_vars = {}
    for var, levels in compressed_variables.items():
        has_levels = bool(levels)
        dims = (
            base_dims if not has_levels else base_dims[:3] + ("level",) + base_dims[3:]
        )
        shape = (
            base_shape
            if not has_levels
            else base_shape[:3] + (len(levels),) + base_shape[3:]
        )
        chunks = (
            base_chunks
            if not has_levels
            else base_chunks[:3] + (len(levels),) + base_chunks[3:]
        )

        data_vars[var] = (
            dims,
            da.zeros(shape, dtype=np.float32, chunks=chunks),
        )

    xr.Dataset(data_vars, coords=coords).to_zarr(ofile, mode="w", consolidated=True)


def fast_create_empty_zarr(
    ofile: str,
    dataset,
    members: int,
    steps: int,
    interval: int = 6,  # 6, 12, or 24
    batch: int = 1,
    indices: list | None = None,
):
    n = len(dataset)
    if indices is None:
        indices = np.arange(n, dtype=int)
    else:
        assert len(indices) == n

    time_coord = np.array(
        [dataset.get_time(i) for i in indices],
        dtype="datetime64[ns]",
    )
    pred_td = (np.arange(steps + 1) * np.timedelta64(interval, "h")).astype(
        "timedelta64[ns]"
    )

    lat, lon = dataset.get_lat_lon()
    n_lat, n_lon = len(lat), len(lon)
    number = np.arange(members, dtype=int)

    coords = {
        "time": (("time",), time_coord),
        "prediction_timedelta": (("prediction_timedelta",), pred_td),
        "latitude": (("latitude",), lat),
        "longitude": (("longitude",), lon),
        "number": (("number",), number),
    }

    compressed_variables = compress_variables(dataset.variables)
    if any(len(lv) for lv in compressed_variables.values()):
        max_lv = max(len(lv) for lv in compressed_variables.values())
        coords["level"] = (("level",), np.arange(max_lv, dtype=int))

    coords_ds = xr.Dataset(coords=coords)
    coords_ds.to_zarr(ofile, mode="w")

    with zarr.open_group(ofile, mode="a") as zarr_group:
        for var, levels in compressed_variables.items():
            has_levels = bool(levels)
            shape = (
                (n, members, steps + 1, n_lat, n_lon)
                if not has_levels
                else (n, members, steps + 1, len(levels), n_lat, n_lon)
            )
            chunks = (
                (batch, 1, steps + 1, n_lat, n_lon)
                if not has_levels
                else (batch, 1, steps + 1, len(levels), n_lat, n_lon)
            )
            ds = zarr_group.create_dataset(
                var, shape=shape, chunks=chunks, dtype="f4", fill_value=0.0
            )
            ds.attrs["_ARRAY_DIMENSIONS"] = (
                ["time", "number", "prediction_timedelta", "latitude", "longitude"]
                if not has_levels
                else [
                    "time",
                    "number",
                    "prediction_timedelta",
                    "level",
                    "latitude",
                    "longitude",
                ]
            )


# ----------------------------------------------------------------------------


def create_empty_numpy(
    ofile: str,  # output path
    dataset,  # an object with properties n_channels and img_resolution
    members: int,  # number of ensemble members
    steps: int,  # number of prediction lead-time steps
):
    """Create an empty npy file with the following structure:
    (samples, members, steps, channels, height, width)

    read with `np.load(ofile, mmap_mode="r")`
    """
    np.lib.format.open_memmap(
        ofile,
        dtype=np.float32,
        mode="w+",
        shape=(
            len(dataset),
            members,
            steps + 1,
            dataset.n_target_channels,
            *dataset.img_resolution,
        ),
    )
