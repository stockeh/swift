<p align="center">
  <img src="media/swift-logo-light.png#gh-light-mode-only" alt="light-logo" height="100">
  <img src="media/swift-logo-dark.png#gh-dark-mode-only" alt="dark-logo" height="100">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2509.25631"><img src="https://img.shields.io/badge/arXiv-2509.25631-b31b1b.svg" alt="arXiv"></a>&nbsp;<a href="https://huggingface.co/stockeh/swift-era5-1.4"><img src="https://img.shields.io/badge/%E2%80%8B-Hugging%20Face-FFD21E?logo=huggingface&logoColor=FFD21E" alt="Hugging Face"></a>
</p>

# An Autoregressive Consistency Model for Efficient Weather Forecasting

<p align="center">
  <img src="https://github.com/user-attachments/assets/0de7f7ba-100b-4c7c-b712-2a75189bd404" alt="q700-gif" width="500"><br>
  <em>Q700 6h forecast initialized 2020-08-22T06</em>
</p>

## Setup

**Dataset**: downsampled ERA5 data at 1.40625° spatial resolution (128× 256 pixels) from [WeatherBench2](https://weatherbench2.readthedocs.io) is needed with per-sample h5 files. See paper for data specifics. 

**Checkpoints**: model weights and configs for our Swift and Diffusion model are available on  [HuggingFace](https://huggingface.co/stockeh/swift-era5-1.4). Sample data is available.

**Environment**: get started with cloning the repo and installing swift into a virtual env.
```bash
# load conda env
module load frameworks

# create venv and install library
cd swift
python3 -m venv venv --system-site-packages
source venv/bin/activate
python3 -m pip install --require-virtualenv -e ".[dev]"
```

## Training

Bash scripts to submit distributed pbs jobs can be found under [`scripts/`](scripts), with [`scripts/chain-resume.sh`](scripts/chain-resume.sh) being the entry point to chain together and resume training on Aurora. These call [`train.py`](src/swift/train.py) with the hydra experiment configs (where hyperparams are set) from the [configs directory](src/swift/configs/experiment). For **pretraining**, we can do:
```bash
# Diffusion (trigflow)
bash chain-resume.sh -s 0 -n 5 -b 1 -e era5-swinv2-1.4-trigflow
# Swift-B (scm)
bash chain-resume.sh -s 0 -n 7 -b 1 -e era5-swinv2-1.4-scm
```

With **finetuning**, we need to modify [`scripts/aurora-general.sh`](scripts/aurora-general.sh) to include the `finetune=multistep` hydra argument and the intervals in [`multistep.yaml`](src/swift/configs/finetune/multistep.yaml) config to have the correct intervals, e.g.,
```yaml
finetune:
  intervals: [
    {steps: 1, kimg: 1500},
    # {steps: 2, kimg: 1500},
    # {steps: 3, kimg: 1000},
    # {steps: 4, kimg: 500},
    # {steps: 8, kimg: 500},
  ]
  name: multistep
```
Thereafter, we can resume from the correct, subsequent resume id as
```bash
# Swift
bash chain-resume.sh -s 8 -n 1 -b 1 -e era5-swinv2-1.4-scm
```

## Inference

For simplicity, we run inference within one or more compute node(s) by calling [`generate.py`](src/swift/generate.py) within our virtual environment. Its important to first initialize [`ezpz`](https://github.com/saforem2/ezpz/). For example, to generate 12 members with 64 initial conditions for 15 days on 6h intervals we have
```bash
# init venv
module load frameworks
source venv/bin/activate

source <(curl -s https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh)
ezpz_setup_env

# run generation
launch python -m swift.generate \
  --input results/era5-swinv2-1.4-scm/011 \
  --checkpoint checkpoint-020000 \
  --members 12 \
  --steps 60 \
  --samples 64 \
  --interval 6
```

## BibTeX

If you find this repo useful in your research, please consider citing our paper:

```latex
@misc{stock2025swift,
  title         = {Swift: An Autoregressive Consistency Model for Efficient Weather Forecasting},
  author        = {Stock, Jason and Arcomano, Troy and Kotamarthi, Rao},
  year          = {2025},
  eprint        = {2509.25631},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2509.25631}
}
```
