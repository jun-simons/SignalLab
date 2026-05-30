from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DSPWarning:
    code: str
    message: str
    severity: str = "warning"


def check_frequency_below_nyquist(freq: float, sample_rate: float, label: str = "frequency") -> list[DSPWarning]:
    nyquist = sample_rate / 2
    if freq <= 0:
        return [
            DSPWarning(
                code="non_positive_frequency",
                message=f"{label} must be positive; got {freq}.",
                severity="error",
            )
        ]
    if freq >= nyquist:
        return [
            DSPWarning(
                code="above_nyquist",
                message=f"{label}={freq:g} Hz is at or above Nyquist={nyquist:g} Hz.",
                severity="error",
            )
        ]
    if freq > 0.8 * nyquist:
        return [
            DSPWarning(
                code="near_nyquist",
                message=f"{label}={freq:g} Hz is close to Nyquist={nyquist:g} Hz.",
            )
        ]
    return []


def check_basic_signal_health(data: np.ndarray, sample_rate: float) -> list[DSPWarning]:
    warnings: list[DSPWarning] = []

    if not np.all(np.isfinite(data)):
        warnings.append(
            DSPWarning(
                code="non_finite",
                message="Signal contains NaN or infinite values.",
                severity="error",
            )
        )

    if len(data) < 8:
        warnings.append(
            DSPWarning(
                code="very_short_signal",
                message="Signal has very few samples; spectral estimates may be meaningless.",
            )
        )

    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    if peak > 1.0:
        warnings.append(
            DSPWarning(
                code="possible_clipping",
                message=f"Signal peak amplitude is {peak:.3g}; audio-style normalized signals may clip.",
            )
        )

    duration = len(data) / sample_rate if sample_rate > 0 else 0.0
    if duration > 0:
        fft_resolution = 1.0 / duration
        if fft_resolution > 20:
            warnings.append(
                DSPWarning(
                    code="low_fft_resolution",
                    message=f"FFT bin spacing is {fft_resolution:.3g} Hz; increase duration for finer spectra.",
                )
            )

    return warnings
