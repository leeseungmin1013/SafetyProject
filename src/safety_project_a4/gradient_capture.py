from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn


@dataclass
class CaptureResult:
    loss: float
    dot_with_utility: float
    reference_norm_sq: float
    utility_norm_sq: float
    bases: dict[str, Tensor]
    spool: "GradientSpool | None"


class GradientSpool:
    """Ephemeral raw tensor files supporting bounded-memory gradient reads."""

    def __init__(self, root: str | Path, label: str, *, chunk_elements: int = 8_388_608) -> None:
        digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:12]
        self.root = Path(root).resolve()
        self.path = self.root / f"{label}-{digest}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.chunk_elements = int(chunk_elements)
        if self.chunk_elements < 1:
            raise ValueError("chunk_elements must be positive")
        self._files: dict[str, tuple[Path, tuple[int, ...], torch.dtype]] = {}

    def write(self, name: str, gradient: Tensor) -> None:
        if not gradient.is_contiguous():
            raise ValueError(f"gradient spool requires contiguous tensor: {name}")
        if gradient.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise TypeError(f"gradient spool does not support {gradient.dtype}")
        index = len(self._files)
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        path = self.path / f"{index:05d}-{digest}.bin"
        flat = gradient.detach().view(-1)
        with path.open("xb") as handle:
            for start in range(0, flat.numel(), self.chunk_elements):
                chunk = flat[start : start + self.chunk_elements].to(device="cpu").contiguous()
                handle.write(chunk.view(torch.uint8).numpy().tobytes())
        self._files[name] = (path, tuple(gradient.shape), gradient.dtype)

    def read_slice(self, name: str, start: int, end: int, *, device: torch.device) -> Tensor:
        path, shape, dtype = self._files[name]
        total = 1
        for dimension in shape:
            total *= dimension
        if start < 0 or end > total or end <= start:
            raise ValueError(f"invalid gradient slice for {name}: {start}:{end}")
        element_size = torch.empty((), dtype=dtype).element_size()
        with path.open("rb") as handle:
            handle.seek(start * element_size)
            raw = bytearray(handle.read((end - start) * element_size))
        if len(raw) != (end - start) * element_size:
            raise OSError(f"truncated gradient spool: {name}")
        return torch.frombuffer(raw, dtype=dtype).to(device=device)

    def read(self, name: str, *, device: torch.device) -> Tensor:
        _, shape, _ = self._files[name]
        total = 1
        for dimension in shape:
            total *= dimension
        parts = [
            self.read_slice(name, start, min(start + self.chunk_elements, total), device=device)
            for start in range(0, total, self.chunk_elements)
        ]
        return torch.cat(parts).reshape(shape)

    def cleanup(self) -> None:
        if self.path.parent != self.root:
            raise RuntimeError("gradient spool escaped its root")
        shutil.rmtree(self.path)


def _eligible(name: str, gradient: Tensor, policy: str) -> bool:
    if gradient.ndim != 2:
        return False
    lowered = name.lower()
    if policy == "all_2d":
        return True
    if policy == "linear_2d":
        excluded = ("embed_tokens", "embedding", "lm_head")
        return not any(token in lowered for token in excluded)
    raise ValueError(f"unknown SPF allowlist policy: {policy}")


def _left_basis(gradient: Tensor, rank: int, oversampling: int, niter: int) -> Tensor:
    matrix = gradient.detach().float()
    maximum = min(matrix.shape)
    effective_rank = min(rank, maximum)
    q = min(effective_rank + oversampling, maximum)
    if effective_rank <= 0:
        raise ValueError("rank must be positive")
    # q may exceed the requested rank for randomized-SVD oversampling.
    u, _, _ = torch.svd_lowrank(matrix, q=q, niter=niter)
    return u[:, :effective_rank].contiguous()


def capture_reference_gradient(
    model: nn.Module,
    loss_closure: Callable[[], Tensor],
    *,
    compute_dot: bool,
    compute_bases: bool,
    spool_root: str | Path | None,
    spool_label: str,
    rank: int,
    oversampling: int,
    niter: int,
    allowlist_policy: str,
    chunk_elements: int = 8_388_608,
) -> CaptureResult:
    """Backpropagates a reference loss without changing utility gradients.

    Parameter hooks see the reference gradient and return zeros, so the utility
    gradients already stored in ``parameter.grad`` remain unchanged.
    """

    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named_parameters:
        raise ValueError("model has no trainable parameters")
    device = named_parameters[0][1].device
    dot = torch.zeros((), device=device, dtype=torch.float64)
    norm_sq = torch.zeros((), device=device, dtype=torch.float64)
    utility_norm_sq = torch.zeros((), device=device, dtype=torch.float64)
    bases: dict[str, Tensor] = {}
    spool = GradientSpool(spool_root, spool_label, chunk_elements=chunk_elements) if spool_root is not None else None
    handles = []

    def make_hook(name: str, parameter: nn.Parameter):
        def hook(reference_gradient: Tensor) -> Tensor:
            if not reference_gradient.is_contiguous():
                raise RuntimeError(f"reference gradient is not contiguous: {name}")
            flat_reference = reference_gradient.detach().view(-1)
            if compute_dot and parameter.grad is None:
                raise RuntimeError(f"utility gradient is missing for {name}")
            flat_utility = parameter.grad.detach().view(-1) if compute_dot else None
            for start in range(0, flat_reference.numel(), chunk_elements):
                end = min(start + chunk_elements, flat_reference.numel())
                reference_fp32 = flat_reference[start:end].float()
                norm_sq.add_(reference_fp32.square().sum(dtype=torch.float64))
                if flat_utility is not None:
                    utility_fp32 = flat_utility[start:end].float()
                    dot.add_((reference_fp32 * utility_fp32).sum(dtype=torch.float64))
                    utility_norm_sq.add_(utility_fp32.square().sum(dtype=torch.float64))
            if compute_bases and _eligible(name, reference_gradient, allowlist_policy):
                bases[name] = _left_basis(reference_gradient, rank, oversampling, niter)
            if spool is not None:
                spool.write(name, reference_gradient)
            return torch.zeros_like(reference_gradient)

        return hook

    try:
        for name, parameter in named_parameters:
            handles.append(parameter.register_hook(make_hook(name, parameter)))
        loss = loss_closure()
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise RuntimeError(f"reference loss must be a finite scalar, got {loss}")
        loss.backward()
        return CaptureResult(
            loss=float(loss.detach().item()),
            dot_with_utility=float(dot.item()),
            reference_norm_sq=float(norm_sq.item()),
            utility_norm_sq=float(utility_norm_sq.item()),
            bases=bases,
            spool=spool,
        )
    except Exception:
        if spool is not None:
            spool.cleanup()
        raise
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def apply_spf_projection(
    model: nn.Module, bases: dict[str, Tensor], *, chunk_elements: int = 8_388_608
) -> None:
    missing = set(bases)
    for name, parameter in model.named_parameters():
        basis = bases.get(name)
        if basis is None:
            continue
        missing.discard(name)
        if parameter.grad is None:
            raise RuntimeError(f"cannot project missing utility gradient: {name}")
        if parameter.grad.ndim != 2 or not parameter.grad.is_contiguous():
            raise RuntimeError(f"SPF projection requires a contiguous matrix gradient: {name}")
        rows, columns = parameter.grad.shape
        rows_per_chunk = max(1, chunk_elements // columns)
        coefficients = torch.zeros(
            (basis.shape[1], columns), device=basis.device, dtype=torch.float32
        )
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            utility = parameter.grad[start:end].float()
            coefficients.add_(basis[start:end].T @ utility)
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            utility = parameter.grad[start:end].float()
            projected = utility - basis[start:end] @ coefficients
            parameter.grad[start:end].copy_(projected.to(dtype=parameter.grad.dtype))
    if missing:
        raise RuntimeError(f"captured SPF bases do not match model parameters: {sorted(missing)[:3]}")
