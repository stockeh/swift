import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import ticker
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

print("Loading data...")
ds = xr.open_zarr(
    "/home/jstock/sf/prod/swift/results/era5-swinv2-1.4-scm"
    "/011/output/checkpoint-020000/"
    "output-64i-60s-12m-6h.zarr",
    # 'output-32i-75s-12m-24h.zarr',
    decode_timedelta=False,
    mask_and_scale=False,
)

lats, lons = ds.latitude.values, ds.longitude.values

variables = {
    "t2m (K)": {"key": "2m_temperature"},
    "u10m (m/s)": {"key": "10m_u_component_of_wind"},
    "v10m (m/s)": {"key": "10m_v_component_of_wind"},
    "mslp (hPa)": {"key": "mean_sea_level_pressure", "scale": 0.01},
    "z500 (m²/s²)": {"key": "geopotential", "level": 7},
    "t850 (K)": {"key": "temperature", "level": 10},
    "q700 (g/kg)": {"key": "specific_humidity", "level": 9, "scale": 1000},
    "u850 (m/s)": {"key": "u_component_of_wind", "level": 10},
    "v850 (m/s)": {"key": "v_component_of_wind", "level": 10},
}


T = ds.prediction_timedelta.size
steps_per_day = 4  # 6-hourly --> 4 steps/day

fig = plt.figure(figsize=(9, 4.6), constrained_layout=False)
gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1], hspace=0.01, wspace=0.01)
axs = np.array(
    [
        [fig.add_subplot(gs[r, c], projection=ccrs.PlateCarree()) for c in range(3)]
        for r in range(3)
    ]
).flatten()

data_arrays = []
vmns, vmxs = [], []
for i, (_, var) in enumerate(variables.items()):
    x = ds[var["key"]].isel(time=0, number=0)
    if "level" in var:
        x = x.isel(level=var["level"])
    if "scale" in var:
        x = x * var["scale"]

    x = x.values  # (T, Y, X)
    data_arrays.append(x)

    vmns.append(np.nanmin(x[0]))
    vmxs.append(np.nanmax(x[0]))

print("Finished loading. Preparing figure...")

quadmeshes = []
labels = []
day_text = None

for i, (label, _) in enumerate(variables.items()):
    ax = axs[i]
    ax.coastlines(color="black", linewidth=1)
    qm = ax.pcolormesh(
        lons,
        lats,
        data_arrays[i][0],
        cmap="bone",
        vmin=vmns[i],
        vmax=vmxs[i],
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    quadmeshes.append(qm)

    tbox = ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        fontsize=11,
        ha="left",
        va="top",
        bbox=dict(
            facecolor="white",
            alpha=0.6,
            edgecolor="none",
            pad=2,
            boxstyle="round,pad=0.2",
        ),
    )
    labels.append(tbox)

day_text = axs[0].text(
    0.78,
    0.96,
    "day 0",
    transform=axs[0].transAxes,
    fontsize=12,
    ha="left",
    va="top",
    color="yellow",
)


def update(t):
    for i, qm in enumerate(quadmeshes):
        arr = data_arrays[i][t]
        qm.set_array(arr.ravel())
    day_text.set_text(f"day {t // steps_per_day}")
    return quadmeshes + [day_text]


def frame_generator():
    for t in tqdm(range(T), desc="Rendering frames"):
        yield t


fig.patch.set_alpha(0)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

fps = 10
anim = FuncAnimation(
    fig,
    update,
    frames=frame_generator,
    interval=1000 / fps,
    blit=False,
    cache_frame_data=False,
)
# To display inline in a notebook: from IPython.display import HTML; HTML(anim.to_jshtml())
# To save an mp4 (requires ffmpeg): anim.save("swift_short.mp4", dpi=150, writer="ffmpeg")
# To save a GIF (requires imagemagick): anim.save("swift_short.gif", dpi=100, writer="imagemagick")

anim.save(
    "swift_short.mp4",
    dpi=300,
    writer="ffmpeg",
    fps=fps,
    savefig_kwargs={"facecolor": "none", "pad_inches": 0},
)
