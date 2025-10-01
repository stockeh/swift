import argparse
import os
import time
from glob import glob
from typing import Callable

import ezpz
import numpy as np
import torch
import torch.distributed
import zarr
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from tqdm import tqdm

from swift.data.samplers import AttributeSubset
from swift.generating.factory import sampler_factory
from swift.utils import io
from swift.utils.helpers import get_ckpt_num, run_on_rank0

parser = argparse.ArgumentParser()
# general args
parser.add_argument("--input", type=str, required=True, help="Input directory")
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Checkpoint name (default: latest)"
)
parser.add_argument("--members", type=int, default=1, help="Number of ensemble members")
parser.add_argument("--steps", type=int, default=8, help="Number of prediction steps")
parser.add_argument("--batch", type=int, default=32, help=" Global batch size")
parser.add_argument("--samples", type=int, default=-1, help="Number of samples use")
parser.add_argument(
    "--interval",
    type=int,
    default=6,
    choices=[6, 12, 24],
    help="Interval in hours",
)
# output args
parser.add_argument(
    "--dump", type=str, default="zarr", choices=["zarr", "numpy"], help="Output format"
)

# ----------------------------------------------------------------------------


def rollout_and_save(
    sampler: Callable[..., torch.Tensor],
    dataloader: DataLoader,
    members: int,
    steps: int,
    ofile: str,
    device: torch.device,
    args: argparse.Namespace,
):
    if args.dump == "numpy":
        store = np.lib.format.open_memmap(ofile, mode="r+")
    else:  # zarr
        store = zarr.open_group(ofile, mode="a")

        # get variable indices for variables with 'level' dimensions
        var_indices, index_counter = {}, 0
        for var, levels in io.compress_variables(dataloader.dataset.variables).items():
            if levels:  # multi-level variable
                var_indices[var] = list(
                    range(index_counter, index_counter + len(levels))
                )
                index_counter += len(levels)
            else:  # single-level variable
                var_indices[var] = [index_counter]
                index_counter += 1

    rank = ezpz.get_rank()
    world_size = ezpz.get_world_size()
    residual = getattr(dataloader.dataset, "residual", False)

    with torch.no_grad():
        for m in tqdm(
            range(rank, members, world_size), desc="member", disable=(rank != 0)
        ):
            # new random state for each member
            generator = torch.Generator(device=device).manual_seed(m)
            n_samples = 0
            for (X, _), (idx, _) in tqdm(
                dataloader, desc="batch", leave=False, disable=(rank != 0)
            ):
                X = X[:, : len(dataloader.dataset.variables)].to(
                    device, non_blocking=True
                )
                bs = X.size(0)

                rollout = np.empty((bs, steps + 1, *X.shape[1:]), dtype=np.float32)
                rollout[:, 0] = dataloader.dataset.unstandardize_x(X).cpu().numpy()

                # rollout
                for i in tqdm(
                    range(steps), desc="step", leave=False, disable=(rank != 0)
                ):
                    # add forcings in
                    X = torch.cat(
                        [
                            X,
                            dataloader.dataset.standardize_x(
                                torch.stack(
                                    [
                                        dataloader.dataset.get_forcings(
                                            j + int(i * args.interval // 6)
                                        )
                                        for j in idx
                                    ],
                                    dim=0,
                                ).to(device)
                            ),
                        ],
                        dim=1,
                    )
                    Y = sampler(X, generator=generator)

                    if residual:
                        X_un = dataloader.dataset.unstandardize_x(X)[
                            :, : len(dataloader.dataset.variables)
                        ]
                        Y_un = dataloader.dataset.unstandardize_t(
                            Y, delta=int(args.interval)
                        )
                        X = X_un + Y_un

                        rollout[:, i + 1] = X.cpu().numpy()

                        X = dataloader.dataset.standardize_x(X)
                    else:
                        rollout[:, i + 1] = (
                            dataloader.dataset.unstandardize_x(Y).cpu().numpy()
                        )
                        X = Y

                # save batch's rollout to disk
                if args.dump == "numpy":
                    store[n_samples : n_samples + bs, m] = rollout
                    store.flush()
                else:  # zarr
                    for var, indices in var_indices.items():
                        if len(indices) == 1:  # single-level variable
                            store[var][n_samples : n_samples + bs, m] = rollout[
                                :, :, indices[0]
                            ]
                        else:  # multi-level variable (stack along 'level' dimension)
                            stacked_data = np.stack(
                                [rollout[:, :, i] for i in indices], axis=2
                            )
                            store[var][n_samples : n_samples + bs, m, :] = stacked_data

                n_samples += bs


# ----------------------------------------------------------------------------


def main(args):
    cfg = OmegaConf.load(os.path.join(args.input, ".hydra", "config.yaml"))
    _ = ezpz.setup_torch(backend=cfg.system.torch.backend)

    if ezpz.get_rank() == 0:
        io.log0(OmegaConf.to_yaml(cfg))

    np.random.seed((cfg.seed) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    device = ezpz.get_torch_device(as_torch_device=True)

    io.log0("Loading dataset...")
    dataset: Dataset = instantiate(cfg.data.dataset, split="test", _convert_="object")
    if args.samples == -1:
        indices = list(range(len(dataset)))
        # indices = indices[928:989] # for tropical cyclone (Laura)
    else:
        # indices = list(range(args.samples)) # first n
        indices = np.linspace(
            0,
            len(dataset) - 1 - (args.steps * args.interval // 6),
            num=args.samples,
            dtype=int,
        ).tolist()
    dataset = AttributeSubset(dataset, indices=indices)

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
    ).to(device)

    if args.checkpoint is not None:
        checkpoint_name = args.checkpoint
        if not checkpoint_name.endswith(".pt"):
            checkpoint_name += ".pt"
        ckpt = os.path.join(args.input, "checkpoints", checkpoint_name)
        if not os.path.exists(ckpt):
            raise ValueError(f"Specified checkpoint {ckpt} does not exist")
        ckpt_basename = os.path.basename(checkpoint_name)[:-3]
    else:
        paths = glob(os.path.join(args.input, "checkpoints", "checkpoint*.pt"))
        checkpoints = sorted(paths, key=get_ckpt_num)  # sort by integer not by ASCII
        assert checkpoints, FileNotFoundError(
            f"No checkpoints in {os.path.join(args.input, 'checkpoints')}"
        )
        ckpt = checkpoints[-1]
        ckpt_basename = "latest"
    io.log0(f"Loading checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=device, weights_only=True)
    net.load_state_dict(state["ema"])
    net.eval()
    run_on_rank0(summary, net, depth=3)

    io.log0("Setting up output directory/file...")
    io.log0(
        f"{len(dataset)} initials for {args.steps} steps over {args.members} members"
    )
    odir = os.path.join(args.input, "output", ckpt_basename)
    run_on_rank0(os.makedirs, odir, exist_ok=True)

    filename = f"output-{len(dataset)}i-{args.steps}s-{args.members}m-{args.interval}h"
    if args.dump == "numpy":
        ofile = os.path.join(odir, f"{filename}.npy")
        run_on_rank0(io.create_empty_numpy, ofile, dataset, args.members, args.steps)
    else:  # zarr
        ofile = os.path.join(odir, f"{filename}.zarr")
        run_on_rank0(
            io.fast_create_empty_zarr,
            ofile,
            dataset,
            args.members,
            args.steps,
            interval=args.interval,
            batch=1,
            indices=indices,
        )

    io.log0(f"Setting up sampler...")
    solver_kwargs = {
        "num_steps": 1,  # 10
        "sigma_min": 0.02,
        "sigma_max": 200.0,
        "auxiliary": args.interval / 10.0,
    }
    sampler = sampler_factory("scm", net, **solver_kwargs)  # 2s

    io.log0(f"Rolling out samples...")
    start_t = time.time()
    rollout_and_save(
        sampler,
        dataloader,
        args.members,
        args.steps,
        ofile,
        ezpz.get_torch_device(as_torch_device=True),
        args,
    )
    io.log0(f"Done! Took {time.time() - start_t:.3f} seconds.")

    io.log0("Cleaning up!")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    if ezpz.get_rank() == 0:
        from zarr.convenience import consolidate_metadata

        io.log0("Consolidating Zarr metadata…")
        consolidate_metadata(ofile)

    io.log0(f"Output saved to: {ofile}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
