"""
python -m swift.eval.metrics \
    --truth /flare/SAFS/jstock/data/wb2/1.40625deg_6hr_test.zarr \
    --pred /home/jstock/sf/prod/swift/results/era5-swinv2-1.4-scm/011/output/checkpoint-020000/output-128i-60s-12m-6h.zarr
"""

import argparse
import json
import os
import time

import ezpz
import numpy as np
import torch
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument("--truth", required=True, help="Path to ground-truth")
parser.add_argument("--pred", required=True, help="Path to prediction")

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


def lat_weighted_rmse(pred: torch.Tensor, y: torch.Tensor, vars, lat, log_postfix):
    """(Ensemble Mean) Latitude weighted root mean squared error

    Args:
        pred: [B, (N), V, H, W]
        y: [B, V, H, W]
        vars: list of variable names
        lat: H
    """
    if pred.ndim == 5:
        pred = pred.mean(dim=1)  # ensemble mean
    error = (pred - y) ** 2  # [B, V, H, W]
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H,)
    w_lat = (
        torch.from_numpy(w_lat)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(dtype=error.dtype, device=error.device)
    )
    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )
    return loss_dict


def lat_weighted_crps(pred: torch.Tensor, y: torch.Tensor, vars, lat, log_postfix):
    assert len(pred.shape) == len(y.shape) + 1
    # pred: [B, N, V, H, W] because there are N ensemble members
    # y: [B, V, H, W]

    H, N = pred.shape[-2], pred.shape[1]

    # latitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).to(dtype=pred.dtype, device=pred.device)  # (H, )

    def crps_var(pred_var: torch.Tensor, y_var: torch.Tensor):
        # pred_var: [B, N, H, W]
        # y: [B, H, W]
        # first term: prediction errors
        with torch.no_grad():
            error_term = torch.abs(pred_var - y_var.unsqueeze(1))  # [B, N, H, W]
            error_term = error_term * w_lat.view(1, 1, H, 1)  # [B, N, H, W]
            error_term = torch.mean(error_term)

        # second term: ensemble spread
        with torch.no_grad():
            spread_term = torch.abs(
                pred_var.unsqueeze(2) - pred_var.unsqueeze(1)
            )  # [B, N, N, H, W]
            spread_term = spread_term * w_lat.view(1, 1, 1, H, 1)  # [B, N, N, H, W]
            spread_term = spread_term.mean(dim=(-2, -1))  # [B, N, N]
            spread_term = spread_term.sum(dim=(1, 2)) / (2 * N * (N - 1))  # [B]
            spread_term = spread_term.mean()

        return error_term - spread_term

    loss_dict = {}
    for i, var in enumerate(vars):
        loss_dict[f"crps_{var}_{log_postfix}"] = crps_var(pred[:, :, i], y[:, i])

    return loss_dict


def lat_weighted_spread_skill_ratio(
    pred: torch.Tensor, y: torch.Tensor, vars, lat, log_postfix
):
    assert len(pred.shape) == len(y.shape) + 1
    # pred: [B, N, V, H, W] because there are N ensemble members
    # y: [B, V, H, W]

    rmse_dict = lat_weighted_rmse(pred.mean(dim=1), y, vars, lat, log_postfix)

    H = pred.shape[-2]

    # latitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()
    w_lat = torch.from_numpy(w_lat).to(dtype=pred.dtype, device=pred.device)  # (H, )

    var = torch.var(pred, dim=1)  # [B, V, H, W]
    var = var * w_lat.view(1, 1, H, 1)  # [B, V, H, W]
    spread = var.mean(dim=(-2, -1)).sqrt().mean(dim=0)  # [V]

    loss_dict = {}
    for i, var in enumerate(vars):
        loss_dict[f"ssr_{var}_{log_postfix}"] = (
            spread[i] / rmse_dict[f"rmse_{var}_{log_postfix}"]
        )

    return loss_dict


def _compute_metrics_for_variable(
    pred_tensor, truth_tensor, var_name, lat, lead_time, device, metric_functions
):
    p = torch.from_numpy(pred_tensor).to(device)
    y = torch.from_numpy(truth_tensor).to(device)

    # Ensure correct dimensions for metric calculation
    if p.ndim == 4:  # [B, N, H, W] - add variable dimension
        p = p.unsqueeze(2)  # → [B, N, 1, H, W]
    if y.ndim == 3:  # [B, H, W] - add variable dimension
        y = y.unsqueeze(1)  # → [B, 1, H, W]

    all_metrics = {}
    for metric_func in metric_functions:
        metrics = metric_func(p, y, [var_name], lat, f"{lead_time}h")
        all_metrics.update(metrics)

    return all_metrics


def main(args):
    device = ezpz.get_torch_device(as_torch_device=True)

    ds_truth = xr.open_zarr(args.truth).chunk({"time": -1})
    ds_pred = xr.open_zarr(args.pred, decode_timedelta=True, mask_and_scale=False)

    metric_functions = [
        lat_weighted_crps,
        lat_weighted_rmse,
        lat_weighted_spread_skill_ratio,
    ]

    lat = ds_truth.latitude.values
    init_times = ds_pred.time.values
    truth_times = ds_truth.time.values

    time_to_idx = {t: i for i, t in enumerate(truth_times)}
    init_idxs = np.array([time_to_idx[t] for t in init_times])

    print(f"📊 Loading datasets...")
    load_start = time.time()
    truth_arrays = {v: ds_truth[v].values for v in ds_truth.data_vars}
    print(f"   ✓ Truth loaded in {time.time() - load_start:.2f}s")
    pred_arrays = {v: ds_pred[v].values for v in ds_pred.data_vars}
    load_time = time.time() - load_start
    print(f"   ✓ Pred loaded in {load_time:.2f}s\n")

    dt_truth = (truth_times[1] - truth_times[0]).astype("timedelta64[h]").astype(int)

    print(f"🚀 Starting evaluation...")
    eval_start = time.time()
    all_metrics = {}
    for j, delta in enumerate(ds_pred.prediction_timedelta.values):
        lead_h = delta.astype("timedelta64[h]").astype(int)
        offset = lead_h // dt_truth
        tgt_idxs = init_idxs + offset

        for var in ds_pred.data_vars:
            p_full = pred_arrays[var]
            t_full = truth_arrays[var]

            if var in PRESSURE_LEVEL_VARS:
                p_block = p_full[:, :, j, ...]  # → (B, N, L, H, W)
                t_block = t_full[tgt_idxs, ...]  # → (B,   L, H, W)

                for lvl_idx, pressure in enumerate(DEFAULT_PRESSURE_LEVELS):
                    p_arr = p_block[..., lvl_idx, :, :]  # (B, N, H, W)
                    t_arr = t_block[..., lvl_idx, :, :]  # (B,   H, W)
                    name = f"{var}_{pressure}"
                    m = _compute_metrics_for_variable(
                        p_arr, t_arr, name, lat, lead_h, device, metric_functions
                    )
                    all_metrics.update(m)

            else:
                p_arr = p_full[:, :, j, :, :]  # → (B, N, H, W)
                t_arr = t_full[tgt_idxs, :, :]  # → (B,   H, W)
                m = _compute_metrics_for_variable(
                    p_arr, t_arr, var, lat, lead_h, device, metric_functions
                )
                all_metrics.update(m)

        print(f"\n=== Lead time: {lead_h} h ===")
        for nm, val in all_metrics.items():
            if nm.endswith(f"_{lead_h}h") and any(
                key in nm for key in ["geopotential_500", "2m_temperature"]
            ):
                print(f"{nm}: {val.item():.4f}")

    eval_time = time.time() - eval_start

    pred_parent_dir = os.path.dirname(args.pred)
    output_file = os.path.join(pred_parent_dir, "evaluation_metrics.json")

    print(f"\n📁 Writing results to: {output_file}")

    # Restructure metrics by metric type -> lead time -> variable
    structured_metrics = {}

    for key, value in all_metrics.items():
        key_parts = key.split("_")

        # Extract metric type (rmse, crps, ssr)
        metric_type = key_parts[0]
        # Extract lead time (last part ending with 'h')
        lead_time = key_parts[-1][:-1]
        # Extract variable name
        var_name = "_".join(key_parts[1:-1])

        if metric_type not in structured_metrics:
            structured_metrics[metric_type] = {}
        if lead_time not in structured_metrics[metric_type]:
            structured_metrics[metric_type][lead_time] = {}

        structured_metrics[metric_type][lead_time][var_name] = (
            value.item() if hasattr(value, "item") else float(value)
        )

    # Add metadata to the output
    output_data = {
        "metadata": {
            "prediction_file": os.path.abspath(args.pred),
            "truth_file": os.path.abspath(args.truth),
            "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "timestamp": time.time(),
        },
        "metrics": structured_metrics,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"🎯 Evaluation Summary:")
    print(f"   📥 Data loading: {load_time:.2f}s")
    print(f"   🔄 Evaluation: {eval_time:.2f}s")
    print(f"   ⏰ Total runtime: {load_time + eval_time :.2f}s")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
