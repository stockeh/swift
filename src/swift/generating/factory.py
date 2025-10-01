from typing import Callable

import torch

from swift.generating.diffusion import DiffusionSampler


def sampler_factory(
    mode: str,
    net: torch.nn.Module,
    denoise_dtype: torch.dtype = torch.float32,
    **solver_kwargs,
) -> Callable[..., torch.Tensor]:
    """Factory to return a sampler function based on the mode.

    Args:
        mode: Solver type ("edm" or "dpm")
        net: Neural network model
        denoise_dtype: Data type for denoising operations
        **solver_kwargs: Solver-specific parameters

    Returns:
        Sampler function

    Usage:
        sampler = sampler_factory("edm", net, num_steps=20, sigma_min=0.03)
    """
    O = DiffusionSampler(net)

    if mode == "edm":

        def sampler(
            X: torch.Tensor, generator: torch.Generator, *args, **kwargs
        ) -> torch.Tensor:
            return O.edm_sampler(
                latents=torch.randn(
                    (X.shape[0], net.img_channels, *net.img_resolution),
                    generator=generator,
                    device=X.device,
                ),
                condition=X,
                denoise_dtype=denoise_dtype,
                **solver_kwargs,
            )

    elif mode == "scm":

        def sampler(
            X: torch.Tensor, generator: torch.Generator, *args, **kwargs
        ) -> torch.Tensor:
            return O.scm_solver(
                latents=torch.randn(
                    (X.shape[0], net.img_channels, *net.img_resolution),
                    generator=generator,
                    device=X.device,
                ),
                condition=X,
                denoise_dtype=denoise_dtype,
                **solver_kwargs,
            )

    elif mode == "2s":

        def sampler(
            X: torch.Tensor, generator: torch.Generator, *args, **kwargs
        ) -> torch.Tensor:
            return O.dpm_solver_2s(
                latents=torch.randn(
                    (X.shape[0], net.img_channels, *net.img_resolution),
                    generator=generator,
                    device=X.device,
                ),
                condition=X,
                denoise_dtype=denoise_dtype,
                **solver_kwargs,
            )

    elif mode == "dpm":

        def sampler(
            X: torch.Tensor, generator: torch.Generator, *args, **kwargs
        ) -> torch.Tensor:
            return O.dpm_solver(
                latents=torch.randn(
                    (X.shape[0], net.img_channels, *net.img_resolution),
                    generator=generator,
                    device=X.device,
                ),
                condition=X,
                denoise_dtype=denoise_dtype,
                **solver_kwargs,
            )

    else:
        raise ValueError(f"Unknown solver mode: {mode}")

    return sampler
