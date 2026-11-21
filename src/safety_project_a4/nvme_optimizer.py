from __future__ import annotations

import hashlib
import math
import os
import shutil
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import Optimizer


ChunkProbe = Callable[[str, int, int, Iterator[tuple[str, Tensor]]], None]
ChunkGradient = Callable[[str, int, int, torch.device], Tensor]


class NVMeInspectableAdamW(Optimizer):
    """FP32 AdamW moments on disk, with one parameter chunk resident at a time.

    The state files are transient run state, not resumable checkpoints. A failed
    run must restart because a step can be interrupted between chunk writes.
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        state_root: str | Path,
        run_id: str,
        chunk_elements: int,
        lr: float,
        betas: tuple[float, float] = (0.5, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        named = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
        if not named:
            raise ValueError("no trainable parameters")
        if chunk_elements < 1:
            raise ValueError("chunk_elements must be positive")
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be one safe path component")
        self._names = {id(parameter): name for name, parameter in named}
        self.chunk_elements = int(chunk_elements)
        self.state_root = Path(state_root).resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.state_root / run_id
        self.state_dir.mkdir(exist_ok=False)
        self._files: dict[int, Path] = {}
        self.bytes_read = 0
        self.bytes_written = 0
        super().__init__(
            [parameter for _, parameter in named],
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )

    def _file_for(self, parameter: Tensor) -> Path:
        key = id(parameter)
        if key not in self._files:
            name = self._names[key]
            digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
            path = self.state_dir / f"{len(self._files):05d}-{digest}.bin"
            with path.open("xb") as handle:
                handle.truncate(parameter.numel() * 8)
            self._files[key] = path
        return self._files[key]

    @staticmethod
    def _read_state(handle, offset: int, count: int, device: torch.device) -> Tensor:
        handle.seek(offset)
        raw = bytearray(handle.read(count * 4))
        if len(raw) != count * 4:
            raise OSError("truncated NVMe optimizer state")
        return torch.frombuffer(raw, dtype=torch.float32).to(device=device)

    @staticmethod
    def _write_state(handle, offset: int, value: Tensor) -> None:
        raw = value.detach().contiguous().to(device="cpu").numpy().tobytes()
        handle.seek(offset)
        if handle.write(raw) != len(raw):
            raise OSError("short NVMe optimizer state write")

    @torch.no_grad()
    def step(
        self,
        closure=None,
        *,
        probe_callback: ChunkProbe | None = None,
        no_spf_gradient_provider: ChunkGradient | None = None,
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
                if parameter.grad.is_sparse or not parameter.is_contiguous() or not parameter.grad.is_contiguous():
                    raise RuntimeError("NVMe AdamW requires dense contiguous parameters and gradients")
                name = self._names[id(parameter)]
                gradient = parameter.grad.detach().view(-1)
                flat_parameter = parameter.detach().view(-1)
                state = self.state[parameter]
                step = int(state.get("step", 0)) + 1
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                state_path = self._file_for(parameter)
                state_size = parameter.numel() * 4
                with state_path.open("r+b", buffering=0) as handle:
                    for start in range(0, parameter.numel(), self.chunk_elements):
                        end = min(start + self.chunk_elements, parameter.numel())
                        count = end - start
                        moment = self._read_state(handle, start * 4, count, parameter.device)
                        variance = self._read_state(handle, state_size + start * 4, count, parameter.device)
                        self.bytes_read += count * 8
                        grad = gradient[start:end].float()
                        before = flat_parameter[start:end].float().clone()

                        if probe_callback is None:
                            moment.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                            variance.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                            denominator = variance.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                            update = moment.div(bias_correction1).div_(denominator).mul_(-lr)
                            before.mul_(1.0 - lr * weight_decay).add_(update)
                            flat_parameter[start:end].copy_(before.to(dtype=parameter.dtype))
                        else:
                            completed = False

                            def stages() -> Iterator[tuple[str, Tensor]]:
                                nonlocal completed
                                if no_spf_gradient_provider is None:
                                    raw_grad = grad
                                else:
                                    raw_grad = no_spf_gradient_provider(name, start, end, parameter.device).float()
                                    if raw_grad.numel() != count:
                                        raise ValueError(f"No-SPF gradient chunk size mismatch for {name}")
                                raw_m = moment.mul(beta1).add(raw_grad, alpha=1.0 - beta1)
                                raw_v = variance.mul(beta2).addcmul(
                                    raw_grad, raw_grad, value=1.0 - beta2
                                )
                                raw_denominator = raw_v.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                                no_spf = raw_m.div(bias_correction1).div_(raw_denominator).mul_(-lr)
                                no_spf.add_(before, alpha=-lr * weight_decay)
                                yield "d_no_spf_adamw", no_spf
                                del raw_m, raw_v, raw_denominator, no_spf, raw_grad

                                moment.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                                variance.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                                yield "d_raw", grad.mul(-lr)
                                yield "d_m", moment.mul(-lr / bias_correction1)
                                denominator = variance.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                                d_pre = moment.div(bias_correction1).div(denominator).mul_(-lr)
                                yield "d_pre", d_pre
                                d_adamw = d_pre.add(before, alpha=-lr * weight_decay)
                                yield "d_adamw", d_adamw
                                m_reset = grad.mul(-lr * (1.0 - beta1) / bias_correction1).div_(denominator)
                                yield "d_m_reset", m_reset
                                del m_reset
                                v_reset_denominator = grad.abs().mul_(
                                    math.sqrt((1.0 - beta2) / bias_correction2)
                                ).add_(eps)
                                v_reset = moment.div(bias_correction1).div_(v_reset_denominator).mul_(-lr)
                                yield "d_v_reset", v_reset
                                del v_reset, v_reset_denominator, denominator
                                flat_parameter[start:end].copy_((before + d_adamw).to(dtype=parameter.dtype))
                                actual = flat_parameter[start:end].float().clone().sub_(before)
                                yield "d_actual", actual
                                completed = True

                            probe_callback(name, start, end, stages())
                            if not completed:
                                raise RuntimeError(f"probe callback did not consume all stages for {name}")

                        self._write_state(handle, start * 4, moment)
                        self._write_state(handle, state_size + start * 4, variance)
                        self.bytes_written += count * 8
                    os.fsync(handle.fileno())
                    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                state["step"] = step
        return loss

    def cleanup(self) -> None:
        if self.state_dir.parent != self.state_root:
            raise RuntimeError("optimizer state directory escaped its root")
        shutil.rmtree(self.state_dir)
