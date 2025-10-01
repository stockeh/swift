import hashlib
import os
import shutil
import warnings
from datetime import datetime
from glob import glob
from typing import Tuple

import ezpz
import hydra
import numpy as np
import torch
import torch.distributed
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler
from torchinfo import summary

from swift.data.samplers import DeltaBatchSampler, InfiniteSampler
from swift.models.abstract import _Shape2D
from swift.models.swinv2 import SwinV2
from swift.training.trainer import Trainer
from swift.utils import io, stats
from swift.utils.helpers import get_ckpt_num

try:
    import wandb  # needs cli login
except (ImportError, ModuleNotFoundError):
    wandb = None

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)

if "HYDRA_RUN_ID" not in os.environ:
    os.environ["HYDRA_RUN_ID"] = datetime.now().strftime("%Y%m%d_%H%M%S")


def string_to_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (1 << 31)


def resume_setup(cfg: DictConfig) -> Tuple[DictConfig, str | None]:
    if cfg.resume is None:
        return cfg, None

    finetune = cfg.get("finetune", None)

    run_dir = cfg.resume
    if not os.path.isdir(run_dir):  # relative experiment path?
        # .../results/experiment-name/000
        # .../results/experiment-name/001
        # w/ cwd = 001 and resume = 000
        run_dir = os.path.join(os.path.dirname(os.getcwd()), cfg.resume)

    assert os.path.isdir(run_dir), FileNotFoundError(f"{run_dir} is not a directory")

    config = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    paths = glob(os.path.join(run_dir, "checkpoints", "checkpoint*.pt"))
    checkpoints = sorted(paths, key=get_ckpt_num)  # sort by integer not by ASCII
    assert checkpoints, FileNotFoundError(
        f"No checkpoints in {os.path.join(run_dir, 'checkpoints')}"
    )
    ckpt = checkpoints[-1]  # latest checkpoint

    if ezpz.get_rank() == 0:
        shutil.copytree(
            os.path.join(run_dir, ".hydra"),
            os.path.join(os.getcwd(), ".hydra"),
            dirs_exist_ok=True,
        )

    if finetune is not None:
        for k, v in finetune.items():
            config[k] = v

        if config.finetune.get("name", None) == "multistep":
            # resume kwargs + new kimg
            config.trainer.total_kimg = int(
                os.path.basename(ckpt).split("-")[1].split(".")[0]
            ) + sum(
                interval["kimg"] for interval in config.finetune.get("intervals", [])
            )
            # keep learning rate constant
            config.trainer.lr_cosine_anneal = False
            config.trainer.checkpoint_ticks = 200
            config.trainer.val_ticks = 50

            # args for debug testing
            # config.data.batch_size = 12
            # config.trainer.kimg_per_tick = 0.1

        if ezpz.get_rank() == 0:
            with open(os.path.join(os.getcwd(), ".hydra", "config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(config))

    io.log0(f"Resuming from {ckpt}")
    return DictConfig(config), ckpt


def distill_setup(cfg: DictConfig, dataset: Dataset) -> Tuple[DictConfig, str | None]:
    if cfg.distill is None:
        return

    run_dir = cfg.distill
    config = DictConfig(OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml")))
    paths = glob(os.path.join(run_dir, "checkpoints", "checkpoint*.pt"))
    checkpoints = sorted(paths, key=get_ckpt_num)  # sort by integer not by ASCII
    assert checkpoints, FileNotFoundError(
        f"No checkpoints in {os.path.join(run_dir, 'checkpoints')}"
    )
    ckpt = checkpoints[-1]  # latest checkpoint

    if ezpz.get_rank() == 0:
        print(f"Loading distillation model: {ckpt}")

    net_pretrained: torch.nn.Module = instantiate(
        config.precond,
        model_config=config.model,
        img_resolution=dataset.img_resolution,
        img_channels=dataset.n_target_channels,
        condition_channels=dataset.n_condition_channels,
        _recursive_=False,  # to not instantiate the model twice
        _convert_="object",
    )
    net_pretrained.eval().to(ezpz.get_torch_device())

    state = torch.load(ckpt, map_location=ezpz.get_torch_device(), weights_only=True)
    net_pretrained.load_state_dict(state["ema"])

    return net_pretrained


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    _ = ezpz.setup_torch(backend=cfg.system.torch.backend)
    stats.init_multiprocessing(
        rank=ezpz.get_rank(),
        sync_device=torch.device(ezpz.get_torch_device_type()),
    )

    cfg, ckpt = resume_setup(cfg)
    if cfg.get("finetune", None) is not None and ckpt is None:
        io.log0("ERROR: must have resume path to finetune")
        return

    if ezpz.get_rank() == 0:
        io.log0(OmegaConf.to_yaml(cfg))
        io.log0(f"Results directory: {os.getcwd()}")
        if wandb is not None and not os.environ.get("WANDB_DISABLED", False):
            _ = ezpz.setup_wandb(project_name="swift", config=cfg)

    cfg.seed = cfg.seed + string_to_int(os.environ["HYDRA_RUN_ID"])

    np.random.seed((cfg.seed * ezpz.get_world_size() + ezpz.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    if torch.cuda.is_available():
        # reccomended true fixed input size
        torch.backends.cudnn.benchmark = cfg.system.torch.benchmark
        # true on Ampere GPUs (NV V100, A100, or H100)
        torch.backends.cudnn.allow_tf32 = cfg.system.torch.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = cfg.system.torch.allow_tf32
        torch.set_float32_matmul_precision(
            cfg.system.torch.set_float32_matmul_precision
        )
        # for Mixed-Precision Training
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            cfg.system.torch.amp_type == "float16"
        )

    io.log0("Loading dataset...")
    dataset: Dataset = instantiate(cfg.data.dataset, _convert_="object")
    dataset_sampler: Sampler = InfiniteSampler(
        dataset=dataset,
        rank=ezpz.get_rank(),
        num_replicas=ezpz.get_world_size(),
        shuffle=True,
        seed=cfg.seed,
    )
    # if finetune, use DeltaBatchSampler
    common_loader_kwargs = dict(
        dataset=dataset,
        pin_memory=True,
        num_workers=cfg.data.data_workers,
        prefetch_factor=(2 if cfg.data.data_workers > 0 else None),
        persistent_workers=True,
    )

    local_batch_size = cfg.data.batch_size // ezpz.get_world_size()
    if cfg.get("finetune", None) is not None:
        batch_sampler = DeltaBatchSampler(
            sampler=dataset_sampler,
            batch_size=local_batch_size,
            intervals=dataset.intervals,
            seed=cfg.seed,
        )
        loader_kwargs = {
            **common_loader_kwargs,
            "batch_sampler": batch_sampler,
        }
    else:
        loader_kwargs = {
            **common_loader_kwargs,
            "sampler": dataset_sampler,
            "batch_size": local_batch_size,
        }

    dataloader = DataLoader(**loader_kwargs)

    io.log0("Constructing network...")
    net: torch.nn.Module = instantiate(
        cfg.precond,
        model_config=cfg.model,
        img_resolution=dataset.img_resolution,
        img_channels=dataset.n_target_channels,
        condition_channels=dataset.n_condition_channels,
        _recursive_=False,  # to not instantiate the model twice
        _convert_="object",
    )
    net.train().requires_grad_(True).to(ezpz.get_torch_device())

    # Initialize variables needed for roll out validation
    if (
        cfg.trainer.val_ticks is not None
        and "era5" in str(cfg.data.dataset._target_).lower()
    ):
        val_cfg = {
            "_target_": "swift.data.era5.ERA5RollOutDataset",
            "split": "val",
            "root": cfg.data.dataset.root,
            "variables": cfg.data.dataset.variables,
            "forcings": cfg.data.dataset.forcings,
            "interval": cfg.trainer.val_target_interval,
        }
        if "residual" in cfg.data.dataset:
            val_cfg.update({"residual": cfg.data.dataset.residual})

        val_dataset: Dataset = instantiate(
            DictConfig(val_cfg),
            _convert_="object",
        )
        dataset_sampler: Sampler = InfiniteSampler(
            dataset=val_dataset,
            rank=ezpz.get_rank(),
            num_replicas=ezpz.get_world_size(),
            shuffle=True,
            seed=cfg.seed,
        )
        # iter created in trainer (need dataset + dataloader)
        val_dataloader = DataLoader(
            dataset=val_dataset,
            sampler=dataset_sampler,
            batch_size=cfg.data.val_local_batch_size,
            pin_memory=True,
            num_workers=cfg.data.data_workers,
            prefetch_factor=(2 if cfg.data.data_workers > 0 else None),
            persistent_workers=False,
        )
    else:
        val_dataloader = None

    if ezpz.get_rank() == 0:
        summary(net, depth=3)
        if wandb is not None and wandb.run is not None and cfg.watch_wandb:
            wandb.run.watch(net, log="all")

    io.log0("Constructing optimizer...")
    param_groups = net.parameters()
    optim_help = "using default hydra config"
    if isinstance(net.model, SwinV2):
        optim_target: str = cfg.optimizer.get("_target_", "")

        if optim_target in ("torch.optim.Adam", "torch.optim.AdamW"):
            decay, no_decay = [], []
            for name, p in net.named_parameters():
                if "pos_embed" in name or ("norm" in name and "modulation" not in name):
                    no_decay.append(p)
                else:
                    decay.append(p)
            param_groups = [
                {"params": decay, "weight_decay": cfg.optimizer.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]
            optim_help = "setting weight_decay=0 for embeddings"

        elif optim_target == "swift.training.optimizers.muon.MuonWithAuxAdam":
            muon_params, adam_params = [], []
            for name, p in net.named_parameters():
                if p.ndim >= 2 and "transformer" in name:
                    muon_params.append(p)
                else:
                    adam_params.append(p)

            param_groups = [
                {
                    "params": muon_params,
                    "use_muon": True,
                    "lr": cfg.optimizer.lr,
                    "weight_decay": cfg.optimizer.weight_decay,
                },
                {
                    "params": adam_params,
                    "use_muon": False,
                    "lr": cfg.optimizer.adam_lr,
                    "betas": tuple(cfg.optimizer.adam_betas),
                    "weight_decay": cfg.optimizer.adam_weight_decay,
                    "eps": cfg.optimizer.adam_eps,
                },
            ]
            optim_help = "using MuonWithAuxAdam"

    optimizer = instantiate(cfg.optimizer, param_groups, _convert_="object")
    io.log0(f"Optimizer: {optim_help}\n", optimizer)

    io.log0("Constructing loss function...")
    OmegaConf.set_struct(cfg.loss, False)
    if cfg.loss._target_.endswith("SCMLoss") and cfg.distill is not None:
        cfg.loss.distillation = True
    net_pretrained = distill_setup(cfg, dataset)

    loss_fn = instantiate(
        cfg.loss,
        dataset=dataset,
        _convert_="object",
    ).to(ezpz.get_torch_device())

    batch_flop = cfg.data.batch_size * getattr(net.model, "single_sample_flop", 0)
    trainer: Trainer = instantiate(
        cfg.trainer,
        net=net,
        optimizer=optimizer,
        loss_fn=loss_fn,
        amp_type=cfg.system.torch.amp_type,
        ckpt=ckpt,
        flop_count=batch_flop,
        net_pretrained=net_pretrained,
        solver_kwargs=cfg.get("solver", None),
        finetune_kwargs=cfg.get("finetune", None),
    )

    io.log0("Training...")
    trainer.train(dataloader, val_dataloader)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
