from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalComparison:
    reference: object
    candidate: object

    def _aligned(self) -> tuple[np.ndarray, np.ndarray]:
        if self.reference.sample_rate != self.candidate.sample_rate:
            raise ValueError("Cannot compare signals with different sample rates.")
        n = min(self.reference.n_samples, self.candidate.n_samples)
        return self.reference.data[:n], self.candidate.data[:n]

    @property
    def mse(self) -> float:
        ref, cand = self._aligned()
        return float(np.mean((cand - ref) ** 2))

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.mse))

    @property
    def correlation(self) -> float:
        ref, cand = self._aligned()
        if np.std(ref) == 0 or np.std(cand) == 0:
            return float("nan")
        return float(np.corrcoef(ref, cand)[0, 1])

    @property
    def snr_db(self) -> float:
        ref, cand = self._aligned()
        noise = cand - ref
        signal_power = np.mean(ref**2)
        noise_power = np.mean(noise**2)
        if noise_power == 0:
            return float("inf")
        if signal_power == 0:
            return float("-inf")
        return float(10 * np.log10(signal_power / noise_power))

    def metrics(self) -> dict[str, float]:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "correlation": self.correlation,
            "snr_db": self.snr_db,
        }

    def summary(self) -> str:
        metrics = self.metrics()
        result = (
            "Signal comparison\n"
            f"- RMSE: {metrics['rmse']:.6g}\n"
            f"- MSE: {metrics['mse']:.6g}\n"
            f"- Correlation: {metrics['correlation']:.6g}\n"
            f"- SNR: {metrics['snr_db']:.3f} dB"
        )
        print(result)
        return result
