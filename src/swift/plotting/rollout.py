"""
Example usage:

python -m swift.plotting.rollout \
    --prediction_path=/home/jstock/sf/data/2138k-output-366i-180s-24h.zarr \
    --variable=q700 \
    --name=test \
    --time_start=2020-01-01
"""

import argparse
import os

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FuncAnimation

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prediction_path", type=str, required=True, help="Prediction zarr file"
)
parser.add_argument("--target_path", type=str, required=False, help="Target zarr file")
parser.add_argument("--difference", action="store_true", help="Plot the difference")

# name
parser.add_argument("--name", type=str, default="unet", help="Model name")

VARIABLES = {
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "z500": "geopotential",
    "q700": "specific_humidity",
}

parser.add_argument(
    "--variable",
    type=str,
    choices=list(VARIABLES.keys()),
    default="t2m",
    help="Variable to plot",
)
parser.add_argument("--member", type=int, default=0, help="Ensemble member to plot")
parser.add_argument(
    "--time_start", type=str, default="2020-01-01", help="Start time for evaluation"
)


def animate(data, lats, lons, variable, name, time_start):
    fig = plt.figure(figsize=(4, 2))
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = ax.pcolormesh(
        lons,
        lats,
        data[0],
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        norm=mcolors.Normalize(vmin=data.min(), vmax=data.max()),
    )
    ax.coastlines()
    cbar = plt.colorbar(im, ax=ax, shrink=1, pad=0.02)

    step_text = ax.text(
        70, 85, "Step: 0", ha="left", va="top", weight="bold", fontsize=11, color="w"
    )
    ax.set_title(f"{name}, {variable}", loc="left", fontsize=11)

    generator = lambda data, steps: ((i, frame) for i, frame in enumerate(data[:steps]))

    def update(frame_data):
        i, frame = frame_data
        step_text.set_text(f"Step: {i}")
        im.set_array(frame.ravel())
        return [im, step_text]

    fps = 10
    steps = len(data)
    save_as = f"media/{name}-{variable}.gif"

    ani = FuncAnimation(
        fig,
        update,
        frames=generator(data, steps),
        interval=1000 / fps,
        blit=True,
        cache_frame_data=False,
    )

    fig.tight_layout()
    ani.save(save_as, fps=fps, dpi=300)


def load_data(file, args, coords=False):
    # assert file exists
    assert os.path.exists(file), f"File {file} does not exist"

    ds = xr.open_zarr(file, decode_timedelta=False)
    data = ds[VARIABLES[args.variable]].sel(
        time=np.array(args.time_start, dtype="datetime64[ns]"),
    )

    if "number" in data.dims:  # default first member
        data = data.isel(number=0)

    if args.variable == "z500":
        data = data.sel(level=7)
    elif args.variable == "q700":
        data = data.sel(level=9)
    data = data.values
    print(data.shape)

    if coords:
        lats, lons = ds["latitude"].values, ds["longitude"].values
        return data, lats, lons
    return data


def main(args):
    pred, lats, lons = load_data(args.prediction_path, args, coords=True)
    if args.target_path:
        target = load_data(args.target_path, args)

    animate(pred, lats, lons, args.variable, args.name, args.time_start)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
