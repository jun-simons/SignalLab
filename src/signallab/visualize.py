from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_time(signal, ax=None, max_points: int | None = None):
    if ax is None:
        _, ax = plt.subplots()

    t = signal.time
    y = signal.data

    if max_points is not None and len(y) > max_points:
        idx = np.linspace(0, len(y) - 1, max_points).astype(int)
        t = t[idx]
        y = y[idx]

    ax.plot(t, y)
    ax.set_title(f"Time domain: {signal.name}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    return ax


def plot_fft(signal, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    y = signal.data
    freqs = np.fft.rfftfreq(len(y), d=1 / signal.sample_rate)
    mag = np.abs(np.fft.rfft(y)) / max(len(y), 1)

    ax.plot(freqs, mag)
    ax.set_title(f"FFT magnitude: {signal.name}")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude")
    return ax
