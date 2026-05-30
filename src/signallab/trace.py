from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .warnings import DSPWarning


@dataclass(frozen=True)
class TraceStage:
    index: int
    name: str
    operation: str
    params: dict[str, Any]
    sample_rate: float
    n_samples: int
    duration: float
    rms: float
    peak: float
    mean: float
    warnings: list[DSPWarning] = field(default_factory=list)


@dataclass(frozen=True)
class SignalTrace:
    """Immutable history of signal-processing stages."""

    stages: tuple[TraceStage, ...] = ()

    def add(
        self,
        signal,
        operation: str,
        params: dict[str, Any] | None = None,
        warnings: list[DSPWarning] | None = None,
    ) -> SignalTrace:
        data = signal.data
        stage = TraceStage(
            index=len(self.stages),
            name=signal.name,
            operation=operation,
            params=params or {},
            sample_rate=signal.sample_rate,
            n_samples=signal.n_samples,
            duration=signal.duration,
            rms=float(np.sqrt(np.mean(data**2))) if len(data) else 0.0,
            peak=float(np.max(np.abs(data))) if len(data) else 0.0,
            mean=float(np.mean(data)) if len(data) else 0.0,
            warnings=warnings or [],
        )
        return SignalTrace(stages=self.stages + (stage,))

    def summary(self) -> str:
        lines = []
        for stage in self.stages:
            warn_text = ""
            if stage.warnings:
                warn_text = " | warnings: " + "; ".join(w.message for w in stage.warnings)
            lines.append(
                f"[{stage.index}] {stage.operation} "
                f"name={stage.name!r} "
                f"duration={stage.duration:.4f}s "
                f"rms={stage.rms:.4g} "
                f"peak={stage.peak:.4g}"
                f"{warn_text}"
            )
        result = "\n".join(lines)
        print(result)
        return result

    def warnings(self) -> list[DSPWarning]:
        out: list[DSPWarning] = []
        for stage in self.stages:
            out.extend(stage.warnings)
        return out
