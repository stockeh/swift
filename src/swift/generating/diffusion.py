import numpy as np
import torch


class DiffusionSampler:
    def __init__(self, net):
        super().__init__()
        self.net = net

    @torch.no_grad()
    def edm_sampler(
        self,
        latents,
        condition=None,
        auxiliary=None,
        randn_like=torch.randn_like,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        pipeline_engine=False,
        denoise_dtype=torch.bfloat16,
    ):
        """Proposed EDM sampler (Algorithm 2)."""
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(
            num_steps, dtype=denoise_dtype, device=latents.device
        )
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(denoise_dtype) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            with torch.autocast(x_hat.device.type, enabled=True, dtype=x_hat.dtype):
                if not pipeline_engine:
                    denoised = self.net(x_hat, t_hat, condition, auxiliary).to(
                        denoise_dtype
                    )
                else:
                    ## TODO: PP Prediction
                    ...

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                with torch.autocast(
                    x_next.device.type, enabled=True, dtype=x_next.dtype
                ):
                    if not pipeline_engine:
                        denoised = self.net(x_next, t_next, condition, auxiliary).to(
                            denoise_dtype
                        )
                    else:
                        ## TODO: PP Prediction
                        ...

                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    @torch.no_grad()
    def ablation_sampler(
        self,
        latents,
        condition=None,
        auxiliary=None,
        randn_like=torch.randn_like,
        num_steps=18,
        sigma_min=None,
        sigma_max=None,
        rho=7,
        solver="heun",
        discretization="edm",
        schedule="linear",
        scaling="none",
        epsilon_s=1e-3,
        C_1=0.001,
        C_2=0.008,
        M=1000,
        alpha=1,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
    ):
        """Generalized ablation sampler, representing the superset
        of all sampling methods discussed in the paper."""
        assert solver in ["euler", "heun"]
        assert discretization in ["vp", "ve", "iddpm", "edm"]
        assert schedule in ["vp", "ve", "linear"]
        assert scaling in ["vp", "none"]

        # Helper functions for VP & VE noise level schedules.
        vp_sigma = (
            lambda beta_d, beta_min: lambda t: (
                np.e ** (0.5 * beta_d * (t**2) + beta_min * t) - 1
            )
            ** 0.5
        )
        vp_sigma_deriv = (
            lambda beta_d, beta_min: lambda t: 0.5
            * (beta_min + beta_d * t)
            * (sigma(t) + 1 / sigma(t))
        )
        vp_sigma_inv = (
            lambda beta_d, beta_min: lambda sigma: (
                (beta_min**2 + 2 * beta_d * (sigma**2 + 1).log()).sqrt() - beta_min
            )
            / beta_d
        )
        ve_sigma = lambda t: t.sqrt()
        ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
        ve_sigma_inv = lambda sigma: sigma**2

        # Select default noise level range based on the specified time step discretization.
        if sigma_min is None:
            vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
            sigma_min = {"vp": vp_def, "ve": 0.02, "iddpm": 0.002, "edm": 0.002}[
                discretization
            ]
        if sigma_max is None:
            vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
            sigma_max = {"vp": vp_def, "ve": 100, "iddpm": 81, "edm": 80}[
                discretization
            ]

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)

        # Compute corresponding betas for VP.
        vp_beta_d = (
            2
            * (np.log(sigma_min**2 + 1) / epsilon_s - np.log(sigma_max**2 + 1))
            / (epsilon_s - 1)
        )
        vp_beta_min = np.log(sigma_max**2 + 1) - 0.5 * vp_beta_d

        # Define time steps in terms of noise level.
        step_indices = torch.arange(
            num_steps, dtype=torch.float64, device=latents.device
        )
        if discretization == "vp":
            orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
            sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
        elif discretization == "ve":
            orig_t_steps = (sigma_max**2) * (
                (sigma_min**2 / sigma_max**2) ** (step_indices / (num_steps - 1))
            )
            sigma_steps = ve_sigma(orig_t_steps)
        elif discretization == "iddpm":
            u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
            alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
            for j in torch.arange(M, 0, -1, device=latents.device):  # M, ..., 1
                u[j - 1] = (
                    (u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1)
                    - 1
                ).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[
                ((len(u_filtered) - 1) / (num_steps - 1) * step_indices)
                .round()
                .to(torch.int64)
            ]
        else:
            assert discretization == "edm"
            sigma_steps = (
                sigma_max ** (1 / rho)
                + step_indices
                / (num_steps - 1)
                * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
            ) ** rho

        # Define noise level schedule.
        if schedule == "vp":
            sigma = vp_sigma(vp_beta_d, vp_beta_min)
            sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
            sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
        elif schedule == "ve":
            sigma = ve_sigma
            sigma_deriv = ve_sigma_deriv
            sigma_inv = ve_sigma_inv
        else:
            assert schedule == "linear"
            sigma = lambda t: t
            sigma_deriv = lambda t: 1
            sigma_inv = lambda sigma: sigma

        # Define scaling schedule.
        if scaling == "vp":
            s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
            s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
        else:
            assert scaling == "none"
            s = lambda t: 1
            s_deriv = lambda t: 0

        # Compute final time steps based on the corresponding noise levels.
        t_steps = sigma_inv(self.net.round_sigma(sigma_steps))
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        t_next = t_steps[0]
        x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= sigma(t_cur) <= S_max
                else 0
            )
            t_hat = sigma_inv(self.net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
            x_hat = s(t_hat) / s(t_cur) * x_cur + (
                sigma(t_hat) ** 2 - sigma(t_cur) ** 2
            ).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

            # Euler step.
            h = t_next - t_hat
            with torch.autocast(x_hat.device.type, enabled=True, dtype=x_hat.dtype):
                denoised = self.net(
                    x_hat / s(t_hat), sigma(t_hat), condition, auxiliary
                ).to(torch.float64)
            d_cur = (
                sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)
            ) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
            x_prime = x_hat + alpha * h * d_cur
            t_prime = t_hat + alpha * h

            # Apply 2nd order correction.
            if solver == "euler" or i == num_steps - 1:
                x_next = x_hat + h * d_cur
            else:
                assert solver == "heun"
                with torch.autocast(
                    x_prime.device.type, enabled=True, dtype=x_prime.dtype
                ):
                    denoised = self.net(
                        x_prime / s(t_prime), sigma(t_prime), condition, auxiliary
                    ).to(torch.float64)
                d_prime = (
                    sigma_deriv(t_prime) / sigma(t_prime)
                    + s_deriv(t_prime) / s(t_prime)
                ) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(
                    t_prime
                ) * denoised
                x_next = x_hat + h * (
                    (1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime
                )

        return x_next

    @torch.no_grad()
    def dpm_solver(
        self,
        latents: torch.Tensor,
        condition=None,
        auxiliary=None,
        randn_like=torch.randn_like,
        num_steps=20,
        use_pp=True,
        sigma_min=0.002,
        sigma_max=80,
        rho=7,
        denoise_dtype=torch.float32,
    ) -> torch.Tensor:
        """Mathematically correct, but not for v-prediction."""
        # net may be DDP, so get module attribute
        sigma_data = getattr(self.net, "module", self.net).sigma_data

        # edm time step discretization
        ramp = torch.linspace(0, 1, num_steps, device=latents.device)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        t_steps = torch.atan(sigmas / sigma_data)
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        x_t = latents * sigma_data
        t_prev = None
        pred_prev = None

        for k in range(num_steps):
            s, t = t_steps[k], t_steps[k + 1]
            delta = s - t
            cos_dt, sin_dt = torch.cos(delta), torch.sin(delta)

            # noise prediction at time s
            with torch.autocast(latents.device.type, enabled=True, dtype=denoise_dtype):
                F_s = self.net(
                    x_t / sigma_data, s.repeat(x_t.shape[0]), condition, auxiliary
                )
            if use_pp:  # predict data (dpm-solver++)
                pred = torch.cos(s) * x_t - torch.sin(s) * sigma_data * F_s
                denom = torch.sin(s)
            else:  # predict noise (dpm-solver)
                pred = torch.sin(s) * x_t + torch.cos(s) * sigma_data * F_s
                denom = torch.cos(s)

            first_order_step = cos_dt * x_t - sin_dt * sigma_data * F_s
            if k == 0 or k == num_steps - 1:  # 1st-order step (DDIM)
                x_next = first_order_step
            else:  # 2nd-order correction
                logtan = lambda u: torch.log(torch.tan(torch.clamp(u, 1e-4, 1.569)))
                r_s = (logtan(s) - logtan(t_prev)) / (logtan(s) - logtan(t))
                correction = (sin_dt / (2 * r_s * max(denom, 1e-3))) * (
                    pred_prev - pred
                )

                x_next = first_order_step + (correction if use_pp else -correction)

            # shift for next iteration
            t_prev = s
            pred_prev = pred
            x_t = x_next

        return x_t

    @torch.no_grad()
    def dpm_solver_2s(
        self,
        latents: torch.Tensor,
        condition=None,
        auxiliary=None,
        randn_like=torch.randn_like,
        num_steps: int = 20,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = 1.57,  # pi/2
        S_noise: float = 1.0,
        denoise_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """DPM-Solver++ 2S (2-step, 2nd-order Heun)"""
        sigma_data = getattr(self.net, "module", self.net).sigma_data

        device = latents.device
        log_sig_min = torch.log(torch.tensor(sigma_min, device=device))
        log_sig_max = torch.log(torch.tensor(sigma_max, device=device))
        u = torch.linspace(1, 0, num_steps, device=device)
        tau = torch.exp(log_sig_min + u * (log_sig_max - log_sig_min))
        t_steps = torch.atan(tau / sigma_data)
        t_steps = torch.cat([t_steps, torch.zeros(1, device=device)])  # include t=0
        # t_steps[0] = torch.tensor(torch.pi / 2, device=device)

        x_t = latents * sigma_data
        B = latents.size(0)

        for k in range(num_steps):
            s, t = t_steps[k], t_steps[k + 1]

            # # increase noise temporarily
            # gamma = (
            #     min(S_churn / num_steps, np.sqrt(2) - 1)
            #     if S_min <= s_cur <= S_max
            #     else 0
            # )
            # s = s_cur + gamma * s_cur
            # x_t = x_t + (s**2 - s_cur**2).sqrt() * S_noise * randn_like(x_t)

            delta = t - s
            with torch.autocast(device.type, enabled=True, dtype=denoise_dtype):
                F_s = self.net(x_t / sigma_data, s.repeat(B), condition, auxiliary)

            # Euler
            x_euler = x_t + delta * sigma_data * F_s

            # second-order Heun correction
            if k < num_steps - 1:
                with torch.autocast(device.type, enabled=True, dtype=denoise_dtype):
                    F_t = self.net(
                        x_euler / sigma_data, t.repeat(B), condition, auxiliary
                    )
                x_t = x_t + delta * sigma_data * 0.5 * (F_s + F_t)
            else:
                x_t = x_euler

        return x_t

    @torch.no_grad()
    def scm_solver(
        self,
        latents: torch.Tensor,
        condition=None,
        auxiliary=None,
        randn_like=torch.randn_like,
        num_steps: int = 2,
        intermediates: list[float] | None = None,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        denoise_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Multistep Consistency Sampler with sigma_data scaling (TrigFlow)."""
        device = latents.device
        B = latents.shape[0]
        sigma_data = getattr(self.net, "module", self.net).sigma_data

        if num_steps == 1:
            t_steps = torch.tensor([torch.pi / 2], device=device)
        else:
            log_sig_min = torch.log(torch.tensor(sigma_min, device=device))
            log_sig_max = torch.log(torch.tensor(sigma_max, device=device))
            u = torch.linspace(1, 0, num_steps, device=device)
            tau = torch.exp(log_sig_min + u * (log_sig_max - log_sig_min))
            t_steps = torch.atan(tau / sigma_data)

        t_steps = torch.cat([t_steps, torch.zeros(1, device=device)])  # include t=0

        if num_steps == 2 and intermediates is None:
            t_steps = torch.tensor([t_steps[0], 1.1, 0.0], device=device)
        elif intermediates:
            mids = torch.as_tensor(intermediates, device=device)
            t_steps = torch.cat([t_steps[:1], mids, t_steps[-1:]])

        x_t = latents * sigma_data
        for i, t in enumerate(t_steps[:-1]):
            if i > 0:
                noise = sigma_data * randn_like(x_t)
                x_t = torch.sin(t) * noise + torch.cos(t) * x_t
            with torch.autocast(device.type, enabled=True, dtype=denoise_dtype):
                F_t = self.net(x_t / sigma_data, t.expand(B), condition, auxiliary)
            x_t = torch.cos(t) * x_t - torch.sin(t) * sigma_data * F_t

        return x_t

    @torch.no_grad()
    def scm_solve2(
        self,
        latents: torch.Tensor,
        condition=None,
        auxiliary=None,
        randn_like=torch.randn_like,
        num_steps=2,
        intermediates: list | None = None,
        sigma_min=0.002,
        sigma_max=80,
        denoise_dtype=torch.float32,
    ) -> torch.Tensor:
        """Few step sampler for TrigFlow consistency models."""
        sigma_data = getattr(self.net, "module", self.net).sigma_data

        # log-uniform time step discretization
        log_sig_min = torch.log(torch.as_tensor(sigma_min, device=latents.device))
        log_sig_max = torch.log(torch.as_tensor(sigma_max, device=latents.device))
        u = torch.linspace(1, 0, num_steps, device=latents.device)
        tau = torch.exp(log_sig_min + u * (log_sig_max - log_sig_min))
        t_steps = torch.atan(tau / sigma_data)
        t_steps = torch.cat([t_steps, torch.zeros(1, device=latents.device)])

        if num_steps == 2:  # intermediate from scm paper
            t_steps = torch.tensor([t_steps[0], 1.1, 0], device=latents.device)
        elif intermediates and num_steps > 2:
            mids = torch.as_tensor(intermediates, device=latents.device)
            t_steps = torch.cat([t_steps[:1], mids, t_steps[-1:]])
        num_steps = len(t_steps) - 1

        x_t = latents * sigma_data
        B = latents.size(0)
        for k in range(num_steps):
            s, t = t_steps[k], t_steps[k + 1]

            with torch.autocast(latents.device.type, enabled=True, dtype=denoise_dtype):
                F_s = self.net(
                    x_t / sigma_data,
                    s.repeat(B),
                    condition,
                    auxiliary=auxiliary,
                )
            x_t = torch.cos(s) * x_t - torch.sin(s) * sigma_data * F_s

            if num_steps > 1:
                noise = sigma_data * randn_like(F_s)
                x_t = torch.cos(t) * x_t + torch.sin(t) * noise

        return x_t  # / sigma_data
