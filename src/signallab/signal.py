from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

from .trace import SignalTrace
from .warnings import DSPWarning, check_basic_signal_health, check_frequency_below_nyquist


@dataclass(frozen=True)
class Signal:
    """A sampled one-dimensional signal with metadata and optional trace history."""

    data: np.ndarray
    sample_rate: float
    name: str = "signal"
    trace: SignalTrace | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=float)
        if data.ndim != 1:
            raise ValueError(f"Signal data must be 1D. Got shape {data.shape}.")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        object.__setattr__(self, "data", data)
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    @property
    def n_samples(self) -> int:
        return len(self.data)

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.n_samples) / self.sample_rate

    @property
    def nyquist(self) -> float:
        return self.sample_rate / 2

    @classmethod
    def sine(
        cls,
        freq: float,
        duration: float,
        sample_rate: float,
        amplitude: float = 1.0,
        phase: float = 0.0,
        name: str | None = None,
        trace_enabled: bool = True,
    ) -> Signal:
        initial_warnings = check_frequency_below_nyquist(freq, sample_rate, label="sine frequency")
        t = np.arange(int(duration * sample_rate)) / sample_rate
        data = amplitude * np.sin(2 * np.pi * freq * t + phase)
        sig = cls(
            data=data,
            sample_rate=sample_rate,
            name=name or f"sine_{freq:g}Hz",
            metadata={"kind": "sine", "freq": freq, "amplitude": amplitude, "phase": phase},
        )
        return sig._with_initial_trace(trace_enabled, operation="create_sine", warnings=initial_warnings)

    @classmethod
    def noise(
        cls,
        duration: float,
        sample_rate: float,
        amplitude: float = 1.0,
        seed: int | None = None,
        name: str | None = None,
        trace_enabled: bool = True,
    ) -> Signal:
        rng = np.random.default_rng(seed)
        data = amplitude * rng.standard_normal(int(duration * sample_rate))
        sig = cls(
            data=data,
            sample_rate=sample_rate,
            name=name or "noise",
            metadata={"kind": "noise", "amplitude": amplitude, "seed": seed},
        )
        return sig._with_initial_trace(trace_enabled, operation="create_noise")

    def _with_initial_trace(
        self, trace_enabled: bool, operation: str, warnings: list[DSPWarning] | None = None
    ) -> Signal:
        if not trace_enabled:
            return self
        trace = SignalTrace().add(self, operation=operation, params=self.metadata or {}, warnings=warnings or [])
        return replace(self, trace=trace)

    def _spawn(
        self,
        data: np.ndarray,
        operation: str,
        params: dict[str, Any] | None = None,
        name: str | None = None,
        warnings: list[DSPWarning] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Signal:
        new_metadata = dict(self.metadata or {})
        if metadata:
            new_metadata.update(metadata)

        new_signal = Signal(
            data=data,
            sample_rate=self.sample_rate,
            name=name or self.name,
            trace=self.trace,
            metadata=new_metadata,
        )

        if self.trace is not None:
            new_trace = self.trace.add(
                new_signal,
                operation=operation,
                params=params or {},
                warnings=warnings or [],
            )
            new_signal = replace(new_signal, trace=new_trace)

        return new_signal

    def copy(self, name: str | None = None) -> Signal:
        return self._spawn(self.data.copy(), operation="copy", name=name or self.name)

    def normalize(self, peak: float = 1.0) -> Signal:
        max_abs = float(np.max(np.abs(self.data))) if self.n_samples else 0.0
        if max_abs == 0:
            data = self.data.copy()
        else:
            data = self.data / max_abs * peak
        warnings = check_basic_signal_health(data, self.sample_rate)
        return self._spawn(
            data,
            operation="normalize",
            params={"peak": peak},
            warnings=warnings,
        )

    def add_noise(self, snr_db: float, seed: int | None = None) -> Signal:
        rng = np.random.default_rng(seed)
        signal_power = float(np.mean(self.data**2))
        if signal_power == 0:
            noise_power = 1.0
        else:
            noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.normal(0, np.sqrt(noise_power), size=self.n_samples)
        data = self.data + noise
        warnings = check_basic_signal_health(data, self.sample_rate)
        return self._spawn(
            data,
            operation="add_noise",
            params={"snr_db": snr_db, "seed": seed},
            warnings=warnings,
            metadata={"last_added_snr_db": snr_db},
        )

    def lowpass(self, cutoff: float, order: int = 4, zero_phase: bool = True) -> Signal:
        warnings = check_frequency_below_nyquist(cutoff, self.sample_rate, label="lowpass cutoff")
        sos = scipy_signal.butter(order, cutoff, btype="lowpass", fs=self.sample_rate, output="sos")
        if zero_phase:
            data = scipy_signal.sosfiltfilt(sos, self.data)
        else:
            data = scipy_signal.sosfilt(sos, self.data)
        warnings.extend(check_basic_signal_health(data, self.sample_rate))
        return self._spawn(
            data,
            operation="lowpass",
            params={"cutoff": cutoff, "order": order, "zero_phase": zero_phase},
            warnings=warnings,
            metadata={"last_filter": "lowpass", "cutoff": cutoff, "order": order},
        )

    def highpass(self, cutoff: float, order: int = 4, zero_phase: bool = True) -> Signal:
        warnings = check_frequency_below_nyquist(cutoff, self.sample_rate, label="highpass cutoff")
        sos = scipy_signal.butter(order, cutoff, btype="highpass", fs=self.sample_rate, output="sos")
        if zero_phase:
            data = scipy_signal.sosfiltfilt(sos, self.data)
        else:
            data = scipy_signal.sosfilt(sos, self.data)
        warnings.extend(check_basic_signal_health(data, self.sample_rate))
        return self._spawn(
            data,
            operation="highpass",
            params={"cutoff": cutoff, "order": order, "zero_phase": zero_phase},
            warnings=warnings,
            metadata={"last_filter": "highpass", "cutoff": cutoff, "order": order},
        )

    def am_modulate(self, carrier_freq: float, modulation_index: float = 1.0) -> Signal:
        warnings = check_frequency_below_nyquist(
            carrier_freq,
            self.sample_rate,
            label="AM carrier frequency",
        )
        t = self.time
        normalized = self.data / np.max(np.abs(self.data)) if np.max(np.abs(self.data)) > 0 else self.data
        carrier = np.cos(2 * np.pi * carrier_freq * t)
        data = (1 + modulation_index * normalized) * carrier
        warnings.extend(check_basic_signal_health(data, self.sample_rate))
        return self._spawn(
            data,
            operation="am_modulate",
            params={"carrier_freq": carrier_freq, "modulation_index": modulation_index},
            warnings=warnings,
            metadata={"carrier_freq": carrier_freq, "modulation_index": modulation_index},
            name=f"{self.name}_am",
        )

    def am_demodulate(self, carrier_freq: float) -> Signal:
        warnings = check_frequency_below_nyquist(
            carrier_freq,
            self.sample_rate,
            label="AM demodulation carrier frequency",
        )
        t = self.time
        mixed = 2 * self.data * np.cos(2 * np.pi * carrier_freq * t)
        warnings.extend(check_basic_signal_health(mixed, self.sample_rate))
        return self._spawn(
            mixed,
            operation="am_demodulate",
            params={"carrier_freq": carrier_freq},
            warnings=warnings,
            metadata={"demodulated_carrier_freq": carrier_freq},
            name=f"{self.name}_demod",
        )

    def compare_to(self, reference: Signal):
        from .compare import SignalComparison

        return SignalComparison(reference=reference, candidate=self)

    def plot_time(self, ax=None, max_points: int | None = None):
        from .visualize import plot_time

        return plot_time(self, ax=ax, max_points=max_points)

    def plot_fft(self, ax=None):
        from .visualize import plot_fft

        return plot_fft(self, ax=ax)
