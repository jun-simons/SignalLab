from signallab import Lab


def test_near_nyquist_warning_is_recorded():
    lab = Lab(sample_rate=1000, trace=True)
    x = lab.sine(freq=450, duration=1.0)

    warnings = x.trace.warnings()

    assert any(w.code == "near_nyquist" for w in warnings)
