import os
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
PROJECT_DIR = HERE.parent.parent
CONFIGS_DIR = HERE.joinpath("configs")
