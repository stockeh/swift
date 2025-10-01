"""Script for online validation during training. It does not offer ensemble inference"""

import argparse
import os
from glob import glob
from typing import Callable

import ezpz
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from tqdm import tqdm

from swift.data.samplers import AttributeSubset
from swift.generating.factory import sampler_factory
from swift.utils import io
from swift.utils.helpers import run_on_rank0


def RMSE_rollout(
    sampler: Callable[..., torch.Tensor],
    dataloader: DataLoader,
    dataset: Dataset,
    target_interval: int,
    device: torch.device,
    rng: torch.Generator = None,
    num_batches=None,
    pipeline_engine=False,
):
    """Roll out for online validation without writing to disk. Does not consider ensembles
    Input:
        sampler: sampling function created by sampler_factory from factory.py
        rng: generator for consistent initial noise across data parallel group
    """
    rank = ezpz.get_rank()
    num_interval_per_day = 4
    aggregate_rmse_loss = 0  # sum of 14 day loss aggregating channels

    residual = dataset.residual
    arr_separate_rmse_loss = np.zeros(
        [
            dataset.n_target_channels,
            target_interval // num_interval_per_day + 1,
        ]  # c, n_days + 1 (single step)
    )
    if num_batches is None:
        num_batches = len(dataloader)

    batch_idx = 0
    with torch.no_grad():
        ## Iterate through dataloader
        # T: B days c h w
        pbar = tqdm(desc="batch", leave=False, disable=(rank != 0), total=num_batches)
        lat, _ = dataset.get_lat_lon()
        w_lat = torch.cos(torch.deg2rad(torch.tensor(lat).to(device)))
        w_lat = (w_lat / w_lat.mean())[None, None, :, None]
        while batch_idx < num_batches:
            X, TS, idx = next(dataloader)
            X = X.to(device, non_blocking=True)  # B c h w
            TS = TS.to(device, non_blocking=True)

            ## Roll-out to Target Interval
            for i in tqdm(
                range(target_interval), desc="steps", leave=False, disable=(rank != 0)
            ):
                # add forcings in
                X = torch.cat(
                    [
                        X,
                        dataset.standardize_x(
                            torch.stack(
                                [dataset.get_forcings(j + i) for j in idx], dim=0
                            ).to(device)
                        ),
                    ],
                    dim=1,
                )
                # predict next time step
                Y = sampler(X, generator=rng, pipeline_engine=pipeline_engine)

                # compute rmse every new day
                if (i + 1) % num_interval_per_day == 0 or i == 0:
                    nth_day = (i + 1) // num_interval_per_day
                    Y_un = dataset.unstandardize_t(Y)
                    if residual:  # add in real space, remove forcings
                        Y_un = (
                            dataset.unstandardize_x(X)[:, : len(dataset.variables)]
                            + Y_un
                        )
                    # targets are NOT standardized or residual (ERA5RollOutDataset)
                    T_un = TS[:, nth_day]

                    if pipeline_engine:
                        # TODO: gather wp sp pp results
                        ...  # dist.all_gather()

                    # store rmse loss
                    aggregate_rmse_loss += (
                        torch.sqrt(torch.mean((Y_un - T_un) ** 2)).cpu().numpy()
                    )
                    arr_separate_rmse_loss[:, nth_day] += (
                        torch.sqrt(
                            torch.mean(w_lat * ((Y_un - T_un) ** 2), dim=(0, 2, 3))
                        )
                        .cpu()
                        .numpy()
                    )

                if residual:  # convert in real space
                    X = dataset.unstandardize_x(X)[
                        :, : len(dataset.variables)
                    ] + dataset.unstandardize_t(Y)
                    X = dataset.standardize_x(X)
                else:
                    X = Y

            batch_idx += 1
            pbar.update(1)

    aggregate_rmse_loss /= num_batches
    arr_separate_rmse_loss /= num_batches
    pbar.close()

    return aggregate_rmse_loss, arr_separate_rmse_loss


def main(args):
    cfg = OmegaConf.load(os.path.join(args.input, ".hydra", "config.yaml"))
    _ = ezpz.setup_torch(backend=cfg.system.torch.backend)
    io.log0(OmegaConf.to_yaml(cfg))

    np.random.seed((cfg.seed) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))

    ## Load test split dataset
    # target_inteval roll-out (6-hr itnerval, 14 days => 56)
    target_interval = args.target_interval
    io.log0("Loading test dataset...")
    # make T target_inteval target
    dataset: Dataset = instantiate(
        cfg.data.dataset,
        _target_="swift.data.era5.ERA5RollOutDataset",
        split="test",
        _convert_="object",
        interval=target_interval,
    )
    # last target_interval samples are dropped from __Len__
    n_samples = len(dataset) if args.samples == -1 else args.samples
    import random

    strt_idx = random.randint(0, len(dataset))
    # grab random contiguous time interval
    dataset = AttributeSubset(
        dataset, indices=list(range(strt_idx, strt_idx + n_samples))
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.data.data_workers,
        prefetch_factor=(2 if cfg.data.data_workers > 0 else None),
        persistent_workers=False,
    )

    io.log0("Constructing network...")
    net: torch.nn.Module = instantiate(
        cfg.precond,
        model_config=cfg.model,
        img_resolution=dataset.img_resolution,
        img_channels=dataset.n_target_channels,
        condition_channels=dataset.n_condition_channels,
        sigma_max=float("inf"),  # TODO: fix in precond / hydra .inf
        _recursive_=False,  # to not instantiate the model twice
        _convert_="object",
    )

    ckpt = sorted(glob(os.path.join(args.input, "checkpoints", "checkpoint*.pt")))[-1]
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["ema"])
    net.eval().to(ezpz.get_torch_device())
    run_on_rank0(summary, net, depth=3)
    sampler = sampler_factory("edm", net)

    io.log0(f"Rolling out samples...")
    dev = ezpz.get_torch_device(as_torch_device=True)
    rmse_loss = RMSE_rollout(sampler, dataloader, target_interval, dev)
    print(f"rmse_loss: {rmse_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument("--input", type=str, required=True, help="Input ckpt directory")
    parser.add_argument("--batch", type=int, default=32, help=" Global batch size")
    parser.add_argument("--samples", type=int, default=-1, help="Number of samples use")
    parser.add_argument(
        "--target_interval",
        type=int,
        default=56,
        help="number of 6-hour intervals to predict ahead",
    )

    args = parser.parse_args()

    main(args)
