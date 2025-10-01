import time

import ezpz
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Sampler

from swift.data.era5 import ERA5Dataset
from swift.data.samplers import InfiniteSampler

DEVICE = ezpz.get_torch_device(as_torch_device=True)
DATA_WORKERS = 8
BATCH_SIZE = 256


def hydra_setup():
    cfg = DictConfig(
        {
            "data": {
                "dataset": {
                    "_target_": "swift.era5.ERA5Dataset",
                    "root": "/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df",
                    "variables": [
                        "2m_temperature",
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                    ],
                },
                "batch_size": 256,
                "data_workers": 8,
            }
        }
    )

    dataset: Dataset = instantiate(cfg.data.dataset, _convert_="object")
    dataset_sampler: Sampler = InfiniteSampler(
        dataset=dataset, rank=0, num_replicas=1, seed=1234
    )
    dataloader = iter(
        DataLoader(
            dataset=dataset,
            sampler=dataset_sampler,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            num_workers=DATA_WORKERS,
            prefetch_factor=2,
            persistent_workers=True,
        )
    )

    return dataloader


def native_setup():
    dataset: Dataset = ERA5Dataset(
        "/eagle/MDClimSim/tungnd/data/wb2/5.625deg_1_step_6hr_h5df",
        split="train",
        variables=[
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
    )
    dataset_sampler: Sampler = InfiniteSampler(
        dataset=dataset, rank=0, num_replicas=1, seed=1234
    )
    dataloader = iter(
        DataLoader(
            dataset=dataset,
            sampler=dataset_sampler,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            num_workers=DATA_WORKERS,
            prefetch_factor=2,
            persistent_workers=True,
        )
    )

    return dataloader


def main():
    # dataloader = native_setup()
    dataloader = hydra_setup()

    for i in range(24):
        start = time.time()

        X, T = next(dataloader)
        X = X.to(DEVICE, non_blocking=True)
        T = T.to(DEVICE, non_blocking=True)

        print(f"{i}: loaded {X.shape[0]} samples in {time.time() - start:.2f} sec")


if __name__ == "__main__":
    main()
