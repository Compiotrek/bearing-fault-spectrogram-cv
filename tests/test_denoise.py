import numpy as np
import pytest

from src.denoise import butterworth_bandpass_filter


def test_filter_preserves_shape_and_returns_float32() -> None:
    signal = np.sin(2 * np.pi * 1000 * np.arange(4096) / 12000)

    filtered = butterworth_bandpass_filter(signal)

    assert filtered.shape == signal.shape
    assert filtered.dtype == np.float32


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"sample_rate": 0}, "sample_rate must be positive"),
        ({"sample_rate": -12000}, "sample_rate must be positive"),
        ({"order": 0}, "order must be positive"),
        ({"order": -1}, "order must be positive"),
        ({"lowcut": 0.0}, "lowcut must be positive"),
        ({"lowcut": -1.0}, "lowcut must be positive"),
        ({"highcut": 6000.0}, "Nyquist"),
        ({"highcut": 7000.0}, "Nyquist"),
        ({"lowcut": 1000.0, "highcut": 500.0}, "lower than highcut"),
    ],
)
def test_filter_rejects_invalid_parameters(
    arguments: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        butterworth_bandpass_filter(np.ones(256), **arguments)


def test_filter_reduces_out_of_band_energy() -> None:
    sample_rate = 12000
    time = np.arange(sample_rate) / sample_rate
    low_frequency = np.sin(2 * np.pi * 20 * time)
    in_band = np.sin(2 * np.pi * 1000 * time)
    high_frequency = np.sin(2 * np.pi * 5500 * time)
    signal = low_frequency + in_band + high_frequency

    filtered = butterworth_bandpass_filter(signal)
    original_spectrum = np.abs(np.fft.rfft(signal))
    filtered_spectrum = np.abs(np.fft.rfft(filtered))
    frequencies = np.fft.rfftfreq(signal.size, d=1 / sample_rate)

    def amplitude_at(frequency: float, spectrum: np.ndarray) -> float:
        index = int(np.argmin(np.abs(frequencies - frequency)))
        return float(spectrum[index])

    assert amplitude_at(20, filtered_spectrum) < (
        0.1 * amplitude_at(20, original_spectrum)
    )
    assert amplitude_at(5500, filtered_spectrum) < (
        0.1 * amplitude_at(5500, original_spectrum)
    )
    assert amplitude_at(1000, filtered_spectrum) > (
        0.8 * amplitude_at(1000, original_spectrum)
    )


def test_filter_does_not_mutate_input() -> None:
    signal = np.sin(2 * np.pi * 1000 * np.arange(4096) / 12000)
    original = signal.copy()

    butterworth_bandpass_filter(signal)

    np.testing.assert_array_equal(signal, original)
