from __future__ import annotations

from dataclasses import dataclass

from .signal import Signal


@dataclass
class Lab:
    """A lightweight experiment/session object.

    The Lab stores default experiment settings such as sample rate and whether
    newly created signals should keep automatic trace history.
    """

    sample_rate: float = 44_100
    trace: bool = True

    def sine(
        self,
        freq: float,
        duration: float,
        amplitude: float = 1.0,
        phase: float = 0.0,
        name: str | None = None,
    ) -> Signal:
        return Signal.sine(
            freq=freq,
            duration=duration,
            sample_rate=self.sample_rate,
            amplitude=amplitude,
            phase=phase,
            name=name,
            trace_enabled=self.trace,
        )

    def noise(
        self,
        duration: float,
        amplitude: float = 1.0,
        seed: int | None = None,
        name: str | None = None,
    ) -> Signal:
        return Signal.noise(
            duration=duration,
            sample_rate=self.sample_rate,
            amplitude=amplitude,
            seed=seed,
            name=name,
            trace_enabled=self.trace,
        )
