"""
launch python -m swift.eval.sampler --input results/era5-swinv2-1.4-scm-muon/008 --checkpoint checkpoint-015000.pt
"""

import argparse
import itertools
import os
from glob import glob

import ezpz
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from tqdm import tqdm

from swift.generating.factory import sampler_factory
from swift.utils import io
from swift.utils.helpers import get_ckpt_num, run_on_rank0

parser = argparse.ArgumentParser()
# general args
parser.add_argument("--input", type=str, required=True, help="Input directory")
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Checkpoint name (default: latest)"
)
parser.add_argument("--seed", type=int, default=0, help="Random seed")

# batch size args
parser.add_argument("--batch", type=int, default=60, help="Global batch size")

# hyperparameter sweep args
parser.add_argument(
    "--num-steps",
    type=int,
    nargs="+",
    default=[32, 16, 8, 4, 2, 1],
    help="Number of steps for sampling",
)
parser.add_argument(
    "--sigma-min",
    type=float,
    nargs="+",
    default=[0.02],
    help="Minimum sigma values",
)
parser.add_argument(
    "--sigma-max",
    type=float,
    nargs="+",
    default=[200.0],
    help="Maximum sigma values",
)


# ----------------------------------------------------------------------------


def sample_experiment(net, dataloader, odir, args):
    params = list(itertools.product(args.num_steps, args.sigma_min, args.sigma_max))
    io.log0(f"Running {len(params)} parameter combinations")

    rank = ezpz.get_rank()
    world_size = ezpz.get_world_size()
    device = ezpz.get_torch_device(as_torch_device=True)

    lat, _ = dataloader.dataset.get_lat_lon()
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = (w_lat / w_lat.mean())[None, None, :, None]

    results = []

    for i, (num_steps, sigma_min, sigma_max) in enumerate(params):
        io.log0(
            f"Testing: num_steps={num_steps}, sigma_min={sigma_min}, sigma_max={sigma_max}"
        )

        solver_kwargs = {
            "num_steps": num_steps,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
        }
        sampler = sampler_factory("scm", net, **solver_kwargs)

        sse, total = 0, 0
        generator = torch.Generator(device=device).manual_seed(i)

        for X, T in tqdm(dataloader, desc="batch", leave=False, disable=(rank != 0)):
            X = X.to(device, non_blocking=True)
            T = T.to(device, non_blocking=True)

            Y = sampler(X, generator=generator)

            X = dataloader.dataset.unstandardize_x(X.cpu().numpy())
            Y = dataloader.dataset.unstandardize_t(Y.cpu().numpy())
            T = dataloader.dataset.unstandardize_t(T.cpu().numpy())

            # if residual
            Y = X + Y
            T = X + T

            sse += np.sum(w_lat * (Y - T) ** 2, axis=(0, 2, 3))
            total += Y.shape[0]

        t_sse = torch.tensor(sse, dtype=torch.float64, device=device)
        t_total = torch.tensor(total, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(t_sse, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(t_total, op=torch.distributed.ReduceOp.SUM)

        errors = torch.sqrt(t_sse / (t_total * Y.shape[2] * Y.shape[3]))
        overall_error = torch.mean(errors).item()

        io.log0(f"Per channel error")
        if rank == 0:
            for v, d in zip(dataloader.dataset.variables, errors):
                d = d.item()
                io.log0(f"{v}: {d:.6f}")
                solver_kwargs[f"{v}_error"] = d
            io.log0(f"Overall error: {overall_error}")
            solver_kwargs["overall_error"] = overall_error

            results.append(solver_kwargs)

    if rank == 0 and results:
        path = os.path.join(odir, "sampler_results.csv")
        df = pd.DataFrame(results)
        df.to_csv(path, index=False)
        io.log0(f"Results saved to: {path}")


def main(args):
    cfg = OmegaConf.load(os.path.join(args.input, ".hydra", "config.yaml"))
    _ = ezpz.setup_torch(backend=cfg.system.torch.backend)

    if ezpz.get_rank() == 0:
        io.log0(OmegaConf.to_yaml(cfg))

    np.random.seed(args.seed % (1 << 31))
    torch.manual_seed(args.seed)
    device = ezpz.get_torch_device(as_torch_device=True)

    io.log0("Loading dataset...")
    dataset: Dataset = instantiate(cfg.data.dataset, split="test", _convert_="object")
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch // ezpz.get_world_size(),
        sampler=sampler,
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
    state = torch.load(ckpt, map_location=ezpz.get_torch_device(), weights_only=True)
    net.load_state_dict(state["ema"])

    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[ezpz.get_local_rank()], output_device=ezpz.get_local_rank()
    )
    net.eval()

    run_on_rank0(summary, net, depth=2)

    io.log0("Setting up output directory/file...")
    odir = os.path.join(args.input, "output", ckpt_basename)
    run_on_rank0(os.makedirs, odir, exist_ok=True)

    io.log0("Starting Experiment...")
    sample_experiment(net, dataloader, odir, args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
