import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable, Optional

import ezpz
import psutil
import torch
import torch.distributed
import torch.utils.data
import xarray
from torch.utils.data import DataLoader

from swift.generating.factory import sampler_factory
from swift.training.loss import CRPSLoss, EDMLoss, MSELoss, SCMLoss
from swift.training.validate import RMSE_rollout
from swift.utils import stats

logger = ezpz.get_logger(__name__)

try:
    import intel_extension_for_pytorch as ipex
except (ImportError, ModuleNotFoundError):
    ipex = None


class Trainer:
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        total_kimg: int = 200000,  # n steps, measured in thousands of training images.
        ema_halflife_kimg: int = 500,  # half-life of EMA of model weights.
        ema_rampup_ratio: float = 0.05,  # EMA ramp-up coefficient, None = disable.
        lr_rampup_kimg: int = 10000,  # n learning rate ramp-up.
        lr_min_factor: float = 0.01,  # min factor relative to lr [0,1].
        lr_cosine_anneal: bool = True,  # cosine annealing
        kimg_per_tick: int = 50,  # interval of progress prints.
        checkpoint_ticks: int = 50,  # n dump state, None = disable.
        device: Optional[str | torch.device] = None,
        amp_type: Optional[
            str
        ] = "bfloat16",  # None (float32), "bfloat16", or "float16"
        compile: bool = False,
        ckpt: Optional[str] = None,  # checkpoint to resume from
        flop_count: Optional[int] = None,
        profile: bool = False,  # torch profiler
        val_ticks=50,  # n eval ticks, None = disable.
        val_target_interval: int = 56,  # autoregressive steps, 14 days
        val_variables: list[str] = None,  # variables to validate
        net_pretrained: torch.nn.Module = None,  # pretrained model for distillation
        solver_kwargs: Optional[dict] = None,  # solver-specific parameters
        finetune_kwargs: Optional[dict] = None,  # finetune-specific parameters
    ):
        if device is None:
            device = ezpz.get_torch_device(as_torch_device=True)
        if isinstance(device, str):
            device = torch.device(device)
        self.device: torch.device = device
        assert isinstance(self.device, torch.device)
        self.net = net.to(self.device)
        self.enable_amp = amp_type is not None
        self.amp_type = (
            torch.bfloat16
            if amp_type == "bfloat16"
            else torch.float16 if amp_type == "float16" else None
        )
        self.scaler = torch.GradScaler(  # only for float16 stablility
            self.device.type,
            enabled=(amp_type == "float16" and torch.cuda.is_available()),
        )
        self.ddp = (
            torch.nn.parallel.DistributedDataParallel(
                net,
                device_ids=[ezpz.get_local_rank()],
                static_graph=True,  # true may improve efficiency for static models
            )
            if torch.distributed.is_initialized()
            else net
        )

        # compile if possible
        if compile and hasattr(torch, "compile") and torch.cuda.is_available():
            device_cap = torch.cuda.get_device_capability()
            if device_cap in ((7, 0), (8, 0), (9, 0)):
                self.ddp = torch.compile(self.ddp, mode="max-autotune", fullgraph=False)
            else:
                logger.warning(
                    "GPU is not NVIDIA V100, A100, or H100. Speedup may be lower "
                    "than expected.",
                )

        # EMA init
        self.ema = copy.deepcopy(net).eval().requires_grad_(False)

        # set base lr before resuming from checkpoint
        self.base_lr = [g["lr"] for g in optimizer.param_groups]

        # resume from checkpoint
        if ckpt is not None:
            # TODO: do we need to broadcast file IO?
            state = torch.load(ckpt, map_location=self.device, weights_only=True)
            self.net.load_state_dict(state["net"])
            self.ema.load_state_dict(state["ema"])
            self.scaler.load_state_dict(state["scaler"])
            self.resume_kimg = int(os.path.basename(ckpt).split("-")[1].split(".")[0])
            try:  # TODO: allows for different optims, but doesn't reset the same
                optimizer.load_state_dict(state["optimizer"])
            except ValueError:
                logger.warning(f"Could not load optimizer state, starting fresh.")
        else:
            self.resume_kimg = 0

        # pretrained model for distillation
        if net_pretrained is not None:
            self.net_pretrained = net_pretrained.to(self.device)
            self.net_pretrained.eval()
        else:
            self.net_pretrained = None

        self.history = ezpz.History()
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_rampup_kimg = lr_rampup_kimg
        self.lr_min_factor = lr_min_factor
        self.lr_cosine_anneal = lr_cosine_anneal

        # Online Validation
        self.val_ticks = val_ticks
        self.val_target_interval = val_target_interval
        self.val_variables = val_variables
        self.solver_type = "edm" if isinstance(loss_fn, EDMLoss) else "dpm"
        self.solver_kwargs = solver_kwargs or {}
        self.finetune_kwargs = finetune_kwargs or {}

        if self.finetune_kwargs.get("name", None) == "multistep":
            cum_kimg = self.resume_kimg
            for interval in self.finetune_kwargs["intervals"]:
                cum_kimg += interval["kimg"]
                interval["kimg"] = cum_kimg
            logger.info(self.finetune_kwargs)

        # bookkeeping
        self.total_kimg = total_kimg
        self.ema_halflife_kimg = ema_halflife_kimg
        self.ema_rampup_ratio = ema_rampup_ratio
        self.kimg_per_tick = kimg_per_tick
        self.checkpoint_ticks = checkpoint_ticks
        self.flop_count = flop_count

        self.prof = None
        if profile:
            from torch.profiler import ProfilerActivity, profiler, schedule

            def trace_handler(p):
                summary = p.key_averages().table(row_limit=15)
                logger.info(f"prof summary stats:\n{summary}")
                if ezpz.get_rank() == 0:
                    p.export_chrome_trace("rank0_prof.json")

            if torch.xpu.is_available():
                activity = [ProfilerActivity.CPU, ProfilerActivity.XPU]
            elif torch.cuda.is_available():
                activity = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            else:
                activity = [ProfilerActivity.CPU]
            self.tot_num_steps = 9
            my_schedule = schedule(wait=2, warmup=2, active=5)
            self.prof = profiler.profile(
                activities=activity,
                schedule=my_schedule,
                on_trace_ready=trace_handler,
            )

    def _get_batch(
        self, dataloader: Iterable | torch.utils.data.DataLoader
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, int, float], float]:
        t0 = time.perf_counter()
        (x, t), (idx, delta) = next(dataloader)
        x = x.to(self.device, non_blocking=True)
        t = t.to(self.device, non_blocking=True)
        delta = delta.to(self.device, non_blocking=True)
        return (x, t, idx, delta), time.perf_counter() - t0

    def _forward_step(
        self, x: torch.Tensor, t: torch.Tensor, delta: torch.Tensor, **kwargs: dict
    ) -> tuple[torch.Tensor, float]:
        t0 = time.perf_counter()
        with torch.autocast(
            self.device.type, enabled=self.enable_amp, dtype=self.amp_type
        ):
            loss = self.loss_fn(self.ddp, t, condition=x, auxiliary=delta, **kwargs)
        return loss, time.perf_counter() - t0

    def _backward_step(self, global_nimg: int, loss: torch.Tensor) -> float:
        t0 = time.perf_counter()
        # Update learning rate
        warmup_nimg = self.lr_rampup_kimg * 1000
        if global_nimg < warmup_nimg:  # linear warmup
            progress = global_nimg / warmup_nimg
            for g, base_lr in zip(self.optimizer.param_groups, self.base_lr):
                min_lr = base_lr * self.lr_min_factor
                g["lr"] = min_lr + (base_lr - min_lr) * progress
        elif self.lr_cosine_anneal:  # cosine annealing
            progress_cos = min(
                1.0,
                (global_nimg - warmup_nimg) / (self.total_kimg * 1000 - warmup_nimg),
            )
            for g, base_lr in zip(self.optimizer.param_groups, self.base_lr):
                min_lr = base_lr * self.lr_min_factor
                g["lr"] = min_lr + 0.5 * (base_lr - min_lr) * (
                    1 + math.cos(math.pi * progress_cos)
                )

        self.scaler.scale(loss).backward()  # no-op for bfloat16/None amp_type
        self.scaler.unscale_(self.optimizer)

        # nan/gradient clipping
        for param in self.net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad,
                    nan=0,
                    posinf=1e5,
                    neginf=-1e5,
                    out=param.grad,
                )

        # update step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # EMA update
        ema_halflife_nimg = self.ema_halflife_kimg * 1000
        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(
                ema_halflife_nimg, global_nimg * self.ema_rampup_ratio
            )
        ema_beta = 0.5 ** (self.global_batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(self.ema.parameters(), self.net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        return time.perf_counter() - t0

    def _val_step(
        self, val_loader, val_dataset, cur_tick, global_nimg, val_stats_jsonl
    ) -> None:
        sampler = sampler_factory(self.solver_type, self.ema, **self.solver_kwargs)
        val_agg_rmse, val_arr_sep_rmse = RMSE_rollout(
            sampler,
            val_loader,
            val_dataset,
            self.val_target_interval,
            self.device,
            num_batches=1,  # only run on one batch (per GPU) to save time
        )

        # average across ranks
        val_agg_rmse = torch.tensor(val_agg_rmse).to(self.device)
        torch.distributed.all_reduce(val_agg_rmse, torch.distributed.ReduceOp.SUM)
        val_arr_sep_rmse = torch.tensor(val_arr_sep_rmse).to(self.device)
        torch.distributed.all_reduce(val_arr_sep_rmse, torch.distributed.ReduceOp.SUM)
        val_agg_rmse /= ezpz.get_world_size()
        val_arr_sep_rmse /= ezpz.get_world_size()

        data_variables = val_dataset.variables
        variable_rmse_map = dict(zip(data_variables, val_arr_sep_rmse))

        if self.val_variables is None:
            selected_variables = data_variables
        else:
            selected_variables = [
                var for var in self.val_variables if var in variable_rmse_map
            ]
            if not selected_variables:
                selected_variables = data_variables

        # log val metric to wandb
        n_days = val_arr_sep_rmse.shape[1]  # target interval days (+1 for single step)
        wandb_val_metrics = {}
        wandb_val_metrics["train/kimg"] = int(global_nimg / 1e3)
        for var in selected_variables:
            for day in range(n_days):
                # TODO: fix hard coded label if 6h interval changes
                desc = f"6h" if day == 0 else f"{day}day"
                key = f"val/rmse/{desc}/{var}"
                wandb_val_metrics[key] = variable_rmse_map[var][day].item()

        self.history.update(wandb_val_metrics, precision=4)

        val_metrics = {
            "train/kimg": int(global_nimg / 1e3),
            "val/tick": cur_tick,
            **{
                f"val/rmse/{var}": [x.item() for x in variable_rmse_map[var]]
                for var in selected_variables
            },
            "val/rmse": val_agg_rmse.item(),
        }
        logger.info(val_metrics)
        if ezpz.get_rank() == 0:
            val_stats_jsonl.write(json.dumps(val_metrics) + "\n")
            val_stats_jsonl.flush()

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> xarray.Dataset:
        logger.info(f"Training for {self.total_kimg} kimg...")
        stats_jsonl = None
        val_stats_jsonl = None
        cur_tick = 0
        global_nimg = self.resume_kimg * 1000
        tick_start_nimg = global_nimg

        dt_misc = 0
        start_time = time.perf_counter()
        tick_start_time = start_time

        it_train_loader = iter(train_loader)
        (x, _), _ = next(it_train_loader)
        local_batch_size = x.shape[0]
        assert local_batch_size is not None
        self.global_batch_size = local_batch_size * ezpz.get_world_size()
        i = 0  # update each (fwd + bwd) pass
        j = 0  # update each time we skip a tick

        if self.prof is not None:
            self.prof.start()
            logger.info(f"Profiling for {self.prof.step_num} steps")

        if ezpz.get_rank() == 0 and stats_jsonl is None:
            stats_jsonl = open(os.path.join(os.getcwd(), "stats.jsonl"), "at")
            val_stats_jsonl = open(os.path.join(os.getcwd(), "val_stats.jsonl"), "at")

        if val_loader is not None:
            val_dataset = val_loader.sampler.dataset
            val_loader = iter(val_loader)
        else:
            val_dataset = None

        forward_kwargs = {}
        if self.net_pretrained is not None:
            forward_kwargs["net_pretrained"] = self.net_pretrained
        steps = None

        while True:
            t0_iter = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)

            if self.finetune_kwargs.get("name", None) == "multistep":
                interval = self.finetune_kwargs["intervals"][0]
                if steps is None:
                    steps = interval["steps"]
                    # hack to work with batched sampler and infinite sampler
                    if hasattr(train_loader, "batch_sampler"):
                        train_loader.batch_sampler.sampler.set_offset(interval["steps"])
                    else:
                        train_loader.sampler.set_offset(interval["steps"])
                    it_train_loader = iter(train_loader)  # clears old prefetch queue
                elif global_nimg > interval["kimg"] * 1000:
                    self.finetune_kwargs["intervals"].pop(0)
                    if len(self.finetune_kwargs["intervals"]) > 0:
                        interval = self.finetune_kwargs["intervals"][0]
                        steps = interval["steps"]
                        logger.info(f"Switching to interval {interval}")
                        if hasattr(train_loader, "batch_sampler"):
                            train_loader.batch_sampler.sampler.set_offset(
                                interval["steps"]
                            )
                        else:
                            train_loader.sampler.set_offset(interval["steps"])
                        it_train_loader = iter(
                            train_loader
                        )  # clears old prefetch queue
            else:
                steps = 1

            (x, t, idx, delta), dt_data = self._get_batch(it_train_loader)

            if isinstance(self.loss_fn, SCMLoss):
                forward_kwargs["step"] = global_nimg
            elif isinstance(self.loss_fn, (MSELoss, CRPSLoss)):
                forward_kwargs["steps"] = steps
                forward_kwargs["idx"] = idx
            loss, dt_fwd = self._forward_step(x, t, delta, **forward_kwargs)

            if self.prof is not None:
                self.prof.step()

            dt_bwd = self._backward_step(global_nimg, loss)

            if self.prof is not None and (self.prof.step_num > self.tot_num_steps):
                logger.info("Finished Profiling.")
                self.prof.stop()

            # progress tracking
            i += 1  # note: i = j + tick
            global_nimg += self.global_batch_size
            done = global_nimg >= self.total_kimg * 1000

            if (
                (not done)
                and (cur_tick != 0)
                and (global_nimg < (tick_start_nimg + self.kimg_per_tick * 1000))
            ):
                j += 1
                continue

            if (  # rollout validation
                (self.val_ticks is not None)
                and (val_loader is not None)
                and (cur_tick % self.val_ticks == 0)
            ):
                self._val_step(
                    val_loader, val_dataset, cur_tick, global_nimg, val_stats_jsonl
                )

            tick_end_time = time.perf_counter()
            try:
                peak_mem_gb = ezpz.get_max_memory_allocated(self.device) / 2**30
                reserved_mem_gb = ezpz.get_max_memory_reserved(self.device) / 2**20
            except RuntimeError:
                peak_mem_gb = 0
                reserved_mem_gb = 0

            # logging and maintenance
            ezpz.synchronize(
                torch.device(ezpz.get_torch_device_type(), ezpz.get_local_rank())
            )
            dt_tick = tick_end_time - tick_start_time
            # Count number of iterations (fwd + bwd) since last tick
            nimg_since_last_tick = global_nimg - tick_start_nimg
            iters_since_last_tick = nimg_since_last_tick // self.global_batch_size
            # count number of floating point operations (flop) done since last tick
            flop_since_last_tick = iters_since_last_tick * self.flop_count
            tflops = (flop_since_last_tick / dt_tick) / 1e12

            # sync loss across ranks
            loss_value = loss.detach().clone()
            torch.distributed.all_reduce(loss_value, op=torch.distributed.ReduceOp.SUM)
            loss_value = loss_value.item() / ezpz.get_world_size()

            metrics = {
                "train/tick": cur_tick,
                "train/iter": i,
                "train/jter": j,
                "train/loss": loss_value,
                "train/kimg": int(global_nimg / 1e3),
                "train/tflops": tflops,
                "train/dt/dt": tick_end_time - start_time,
                "train/dt/tick": dt_tick,
                "train/dt/iter": tick_end_time - t0_iter,
                "train/dt/data": dt_data,
                "train/dt/fwd": dt_fwd,
                "train/dt/bwd": dt_bwd,
                "train/dt/misc": dt_misc,
                "train/dt/kimg": 1e3 * dt_tick / (global_nimg - tick_start_nimg),
                "train/mem/cpu": psutil.Process(os.getpid()).memory_info().rss / 2**30,
                "train/mem/gpu": peak_mem_gb,
                "train/mem/reserved": reserved_mem_gb,
                "train/lr": self.optimizer.param_groups[0]["lr"],
            }

            logger.info(
                self.history.update(
                    metrics,
                    # use_wandb=False,
                    precision=4,
                    summarize=True,
                )
                .replace("train/", "")
                .replace("dt/", "")
                .replace("mem/", "")
            )

            _ = [stats.report0(key, val) for key, val in metrics.items()]
            stats.default_collector.update()

            if ezpz.get_rank() == 0:
                stats_jsonl.write(
                    json.dumps({**stats.default_collector.as_dict()}) + "\n"
                )
                stats_jsonl.flush()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            elif torch.xpu.is_available() and ipex is not None:
                ipex.xpu.reset_peak_memory_stats()

            if (
                (self.checkpoint_ticks is not None)
                and (done or cur_tick % self.checkpoint_ticks == 0)
                and cur_tick != 0
                and ezpz.get_rank() == 0
            ):
                logger.info(f"Saving checkpoint @ {cur_tick=}, {global_nimg=}...")
                self._save_checkpoint(global_nimg)

            cur_tick += 1
            tick_start_nimg = global_nimg
            tick_start_time = time.perf_counter()
            dt_misc = tick_start_time - tick_end_time
            if done:
                logger.info(
                    f"Finished training in {(tick_end_time - start_time) / 3600:.2f} hours"
                )
                if stats_jsonl is not None:
                    stats_jsonl.close()
                if val_stats_jsonl is not None:
                    val_stats_jsonl.close()
                dataset = self.history.finalize(
                    dataset_fname="train",
                    save=(ezpz.get_rank() == 0),
                    plot=(ezpz.get_rank() == 0),
                    outdir=Path(os.getcwd()).joinpath("outputs"),
                )
                logger.info(f"{dataset=}")
                return dataset

    def _save_checkpoint(self, cur_nimg):
        """resuming and for inference"""
        state = {
            "ema": self.ema.state_dict(),  #! use for inference
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        path = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(path, exist_ok=True)
        torch.save(
            state,
            os.path.join(path, f"checkpoint-{cur_nimg // 1000:06d}.pt"),
        )
