# Signal Lab

Signal Lab is a Python DSP lab notebook for building, tracing, comparing, and debugging signal-processing pipelines.

It provides a higher-level workflow for learning/experiments than numpy or scipy:

```text
create signal → transform signal → automatically trace stages → compare before/after → catch DSP mistakes
```

## Example

```python
from signallab import Lab, Signal

lab = Lab(sample_rate=10_000, trace=True)

message = lab.sine(freq=20, duration=1.0)

recovered = (
    message
    .am_modulate(carrier_freq=1000)
    .add_noise(snr_db=5, seed=42)
    .am_demodulate(carrier_freq=1000)
    .lowpass(cutoff=80, order=4)
)

recovered.trace.summary()
recovered.compare_to(message).summary()
```

## Core Features

- `Signal` object with sample-rate-aware operations
- Automatic signal tracing
- Before/after signal comparisons
- DSP warnings for common mistakes
- Basic filters
- AM modulation and demodulation
- Time-domain and FFT visualization
- Simple report hooks

## Development Setup

```bash
pip install -e ".[dev]"
pytest
```
