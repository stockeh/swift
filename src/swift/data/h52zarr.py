import argparse
import os
from glob import glob

import dask
import dask.array as da
import h5py
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from swift.utils.io import compress_variables

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True, help="Input directory")
parser.add_argument(
    "-s",
    "--split",
    type=str,
    choices=["train", "val", "test"],
    default="test",
    help="Data split",
)
parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
parser.add_argument("-n", "--name", type=str, required=True, help="Dataset name")


def read_variable(file: str, var: str):
    with h5py.File(file, "r") as f:
        return f["input"][var][()]  # (lat, lon)


def read_compressed_variable(
    file: str, base_var: str, sorted_levels: list[int]
) -> np.ndarray:
    slices = []
    with h5py.File(file, "r") as f:
        for lev in sorted_levels:
            var_name = f"{base_var}_{lev}"
            if var_name not in f["input"]:
                raise KeyError(f"Variable {var_name} not found in file {file}.")
            slices.append(f["input"][var_name][()])
    return np.stack(slices, axis=0)  # (lev, lat, lon)


def main(args):
    split_dir = os.path.join(args.input, args.split)
    assert os.path.exists(split_dir), f"Directory {split_dir} does not exist."
    lat_path = os.path.join(args.input, "lat.npy")
    lon_path = os.path.join(args.input, "lon.npy")
    assert os.path.exists(lat_path), "lat.npy not found."
    assert os.path.exists(lon_path), "lon.npy not found."

    files = sorted(glob(os.path.join(split_dir, "*.h5")))
    assert files, "No .h5 files found in the input directory."
    print(f"Converting {len(files)} ({args.split}) files to Zarr...")

    lat = np.load(lat_path).astype(np.float32)
    lon = np.load(lon_path).astype(np.float32)

    with h5py.File(files[0], "r") as f:
        start_time = np.datetime64(f["input"]["time"][()].decode("utf-8"))
        variables = list(f["input"].keys())
        variables = [
            var
            for var in variables
            if var not in ["soil_temperature_level_1", "volumetric_soil_water_layer_1"]
        ]

    # assumes files are on 6-hour intervals
    time_coord = start_time + np.arange(len(files)) * np.timedelta64(6, "h")
    compressed_variables = compress_variables(variables)
    compressed_variables.pop("time", None)

    n_levels = max((len(levels) for levels in compressed_variables.values()), default=0)
    coords = {
        "time": (("time",), time_coord),
        "latitude": (("latitude",), lat),
        "longitude": (("longitude",), lon),
        "level": (("level",), np.arange(n_levels, dtype=np.int32)),
    }
    nlat, nlon = len(lat), len(lon)

    data_vars = {}
    for var, levels in compressed_variables.items():
        if levels:
            sorted_levels = sorted(levels)
            shape = (len(sorted_levels), nlat, nlon)
            dims = ("time", "level", "latitude", "longitude")
            chunks = (1, len(sorted_levels), nlat, nlon)
            slices = [
                dask.delayed(read_compressed_variable)(file, var, sorted_levels)
                for file in files
            ]
        else:
            shape = (nlat, nlon)
            dims = ("time", "latitude", "longitude")
            chunks = (1, nlat, nlon)
            slices = [dask.delayed(read_variable)(file, var) for file in files]

        data = [da.from_delayed(ds, shape=shape, dtype=np.float32) for ds in slices]
        stacked = da.stack(data, axis=0).rechunk(chunks)
        data_vars[var] = (dims, stacked)

    ofile = os.path.join(args.output, f"{args.name}.zarr")
    os.makedirs(args.output, exist_ok=True)

    with dask.config.set(scheduler="threads"):
        with ProgressBar():
            xr.Dataset(data_vars, coords=coords).to_zarr(
                ofile, mode="w", consolidated=True
            )
    print(f"Zarr store created and populated at {ofile}")


if __name__ == "__main__":
    """
    Usage:

        python -m swift.data.h52zarr \
            -i /lus/flare/projects/SAFS/jstock/data/wb2/1.40625deg_1_step_6hr_h5df \
            -s test \
            -o /lus/flare/projects/SAFS/jstock/data/wb2/ \
            -n 1.40625deg_6hr_test

    """
    args = parser.parse_args()
    main(args)
