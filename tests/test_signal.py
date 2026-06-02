from signallab import Lab


def test_trace_records_stages():
    lab = Lab(sample_rate=1000, trace=True)
    x = lab.sine(freq=10, duration=1.0)
    y = x.add_noise(snr_db=10, seed=0).lowpass(cutoff=50)

    assert y.trace is not None
    assert len(y.trace.stages) == 3
    assert y.trace.stages[0].operation == "create_sine"
    assert y.trace.stages[1].operation == "add_noise"
    assert y.trace.stages[2].operation == "lowpass"


def test_compare_to_self_is_perfect():
    lab = Lab(sample_rate=1000, trace=True)
    x = lab.sine(freq=10, duration=1.0)

    comparison = x.compare_to(x)

    assert comparison.rmse == 0
    assert comparison.snr_db == float("inf")
