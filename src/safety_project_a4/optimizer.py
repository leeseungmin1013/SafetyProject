from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator

import torch
from torch import Tensor
from torch.optim import Optimizer


ProbeCallback = Callable[[str, Tensor, Iterator[tuple[str, Tensor]]], None]
GradientProvider = Callable[[str, torch.device], Tensor]


class InspectableAdamW(Optimizer):
    """AdamW with explicit state dtype and streamed A4 stage callbacks.

    The update follows ``torch.optim.AdamW`` for ``amsgrad=False`` and
    ``maximize=False``. Probe vectors are yielded one at a time, so memory is
    bounded by the largest parameter tensor rather than all A4 stages at once.
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        lr: float,
        betas: tuple[float, float] = (0.5, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        state_dtype: str = "fp32",
    ) -> None:
        named = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
        if not named:
            raise ValueError("no trainable parameters")
        self._parameter_names = {id(parameter): name for name, parameter in named}
        self.state_dtype_policy = state_dtype
        super().__init__(
            [parameter for _, parameter in named],
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )

    def _state_dtype(self, parameter: Tensor) -> torch.dtype:
        return torch.float32 if self.state_dtype_policy == "fp32" else parameter.dtype

    @torch.no_grad()
    def step(
        self,
        closure=None,
        *,
        probe_callback: ProbeCallback | None = None,
        no_spf_gradient_provider: GradientProvider | None = None,
    ):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("InspectableAdamW does not support sparse gradients")
                name = self._parameter_names[id(parameter)]
                grad = parameter.grad.detach()
                state = self.state[parameter]
                state_dtype = self._state_dtype(parameter)
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, dtype=state_dtype)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=state_dtype)
                step = int(state["step"]) + 1
                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]
                state_grad = grad.to(dtype=state_dtype)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                if probe_callback is None:
                    exp_avg.mul_(beta1).add_(state_grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        state_grad, state_grad, value=1.0 - beta2
                    )
                    denominator = exp_avg_sq.float().sqrt().div_(
                        math.sqrt(bias_correction2)
                    ).add_(eps)
                    update = exp_avg.float().div(bias_correction1).div_(denominator).mul_(-lr)
                    parameter_fp32 = parameter.detach().float()
                    parameter_fp32.mul_(1.0 - lr * weight_decay).add_(update)
                    parameter.copy_(parameter_fp32.to(dtype=parameter.dtype))
                    state["step"] = step
                    continue

                def stage_iterator() -> Iterator[tuple[str, Tensor]]:
                    parameter_fp32 = parameter.detach().float()
                    grad_fp32 = grad.float()

                    # Counterfactual No-SPF branch from the same pre-step m/v.
                    if no_spf_gradient_provider is None:
                        no_spf_grad = grad_fp32
                    else:
                        no_spf_grad = no_spf_gradient_provider(name, parameter.device).float()
                    no_spf_state_grad = no_spf_grad.to(dtype=state_dtype)
                    no_spf_m = exp_avg.mul(beta1).add(
                        no_spf_state_grad, alpha=1.0 - beta1
                    )
                    no_spf_v = exp_avg_sq.mul(beta2).addcmul(
                        no_spf_state_grad, no_spf_state_grad, value=1.0 - beta2
                    )
                    no_spf_denominator = no_spf_v.float().sqrt().div_(
                        math.sqrt(bias_correction2)
                    ).add_(eps)
                    d_no_spf = no_spf_m.float().div(bias_correction1).div_(
                        no_spf_denominator
                    ).mul_(-lr).add(parameter_fp32, alpha=-lr * weight_decay)
                    yield "d_no_spf_adamw", d_no_spf
                    del no_spf_m, no_spf_v, no_spf_denominator, d_no_spf

                    # Real SPF path. State mutation happens only here.
                    exp_avg.mul_(beta1).add_(state_grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        state_grad, state_grad, value=1.0 - beta2
                    )
                    state["step"] = step

                    d_raw = grad_fp32.mul(-lr)
                    yield "d_raw", d_raw
                    del d_raw

                    d_m = exp_avg.float().mul(-lr / bias_correction1)
                    yield "d_m", d_m
                    del d_m

                    denominator = exp_avg_sq.float().sqrt().div_(
                        math.sqrt(bias_correction2)
                    ).add_(eps)
                    d_pre = exp_avg.float().div(bias_correction1).div(denominator).mul(-lr)
                    yield "d_pre", d_pre
                    d_adamw = d_pre.add(parameter_fp32, alpha=-lr * weight_decay)
                    yield "d_adamw", d_adamw

                    # Same beta/step/current gradient and v, with m_prev reset.
                    d_m_reset = state_grad.float().mul(
                        -lr * (1.0 - beta1) / bias_correction1
                    ).div_(denominator)
                    yield "d_m_reset", d_m_reset
                    del d_m_reset

                    # Same beta/step/current m, with v_prev reset.
                    v_reset_denominator = state_grad.float().abs().mul_(
                        math.sqrt((1.0 - beta2) / bias_correction2)
                    ).add_(eps)
                    d_v_reset = exp_avg.float().div(bias_correction1).div_(
                        v_reset_denominator
                    ).mul_(-lr)
                    yield "d_v_reset", d_v_reset
                    del d_v_reset, v_reset_denominator, denominator

                    before = parameter.detach().clone()
                    parameter.copy_((parameter_fp32 + d_adamw).to(dtype=parameter.dtype))
                    d_actual = parameter.detach().float().clone().sub_(before.float())
                    yield "d_actual", d_actual

                probe_callback(name, grad, stage_iterator())
        return loss
