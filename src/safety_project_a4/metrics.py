from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping


def norm_scaled_tau(norm_a: float, norm_b: float) -> float:
    return max(1e-12, 1e-6 * norm_a * norm_b)


@dataclass(frozen=True)
class DirectionMetric:
    dot: float
    reference_norm: float
    vector_norm: float
    cosine: float | None
    tau: float
    classification: str

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


def direction_metric(dot: float, reference_norm: float, vector_norm: float) -> DirectionMetric:
    if not all(math.isfinite(value) for value in (dot, reference_norm, vector_norm)):
        return DirectionMetric(dot, reference_norm, vector_norm, None, math.nan, "invalid")
    if reference_norm <= 0.0 or vector_norm <= 0.0:
        return DirectionMetric(dot, reference_norm, vector_norm, None, 1e-12, "zero-norm")
    tau = norm_scaled_tau(reference_norm, vector_norm)
    cosine = dot / (reference_norm * vector_norm)
    if dot > tau:
        classification = "conflict"
    elif dot < -tau:
        classification = "safe"
    else:
        classification = "neutral"
    return DirectionMetric(dot, reference_norm, vector_norm, cosine, tau, classification)


def transition_events(metrics: Mapping[str, DirectionMetric]) -> dict[str, bool]:
    def conflict(name: str) -> bool:
        return metrics[name].classification == "conflict"

    return {
        "spf_residual": conflict("d_raw"),
        "momentum_induced": not conflict("d_raw") and conflict("d_m"),
        "preconditioner_induced": not conflict("d_m") and conflict("d_pre"),
        "decay_induced": not conflict("d_pre") and conflict("d_adamw"),
        "realized_update_residual": not conflict("d_adamw") and conflict("d_actual"),
    }


class MetricAccumulator:
    """Accumulates exact global dots/norms without retaining full vectors."""

    STAGES = (
        "d_raw",
        "d_m",
        "d_pre",
        "d_adamw",
        "d_actual",
        "d_m_reset",
        "d_v_reset",
        "d_no_spf_adamw",
    )

    def __init__(self) -> None:
        self.reference_norm_sq = 0.0
        self.dot = {name: 0.0 for name in self.STAGES}
        self.norm_sq = {name: 0.0 for name in self.STAGES}

    def add_reference_norm_sq(self, value: float) -> None:
        self.reference_norm_sq += value

    def add(self, stage: str, dot: float, norm_sq: float) -> None:
        if stage not in self.dot:
            raise KeyError(stage)
        self.dot[stage] += dot
        self.norm_sq[stage] += norm_sq

    def finalize(self) -> dict[str, DirectionMetric]:
        reference_norm = math.sqrt(max(self.reference_norm_sq, 0.0))
        return {
            name: direction_metric(
                self.dot[name], reference_norm, math.sqrt(max(self.norm_sq[name], 0.0))
            )
            for name in self.STAGES
        }
