from signallab import Lab


def main():
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


if __name__ == "__main__":
    main()
