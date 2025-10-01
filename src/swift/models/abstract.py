from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from omegaconf import ListConfig

_Shape2D = Union[int, tuple[int, int], list[int], ListConfig]


class AbstractNetwork(torch.nn.Module, ABC):
    """All networks should inherit this."""

    def __init__(
        self,
        img_resolution: _Shape2D,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        auxiliary: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("subclass must implement this.")


# ----------------------------------------------------------------------------
# Utility Classes


@dataclass
class Shape2D:
    _shape: _Shape2D

    def __post_init__(self) -> None:
        if isinstance(self._shape, int):
            self.shape = (self._shape, self._shape)
        elif isinstance(self._shape, list):
            self.shape = tuple(self._shape)
        elif isinstance(self._shape, tuple):
            self.shape = tuple(self._shape)
        else:
            raise TypeError(f"Invalid type {type(self._shape)}")
        assert (  # [!] self.shape must be (int, int)
            isinstance(self.shape, tuple)
            and isinstance(self.shape[0], int)
            and isinstance(self.shape[1], int)
        )

        self.height = self.shape[0]
        self.width = self.shape[1]


# ----------------------------------------------------------------------------
# Utility Functions


def get_activation(activation_f: str) -> nn.Module:
    activations = [
        nn.Tanh,
        nn.ReLU,
        nn.LeakyReLU,
        nn.SiLU,
        nn.SELU,
        # ...
    ]
    names = [str(o.__name__).lower() for o in activations]
    try:
        return activations[names.index(str(activation_f).lower())]
    except Exception:
        raise NotImplementedError(f"{activation_f=} is not yet implemented.")
