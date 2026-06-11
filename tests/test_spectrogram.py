import numpy as np
import pytest

from src.spectrogram import (
    compute_log_power_spectrogram,
    resize_spectrogram,
    standardize_spectrogram,
)


def test_compute_log_power_spectrogram_returns_finite_2d_float32() -> None:
    time = np.arange(2048) / 12000
    signal = np.sin(2 * np.pi * 1000 * time)

    spectrogram = compute_log_power_spectrogram(signal)

    assert spectrogram.ndim == 2
    assert spectrogram.dtype == np.float32
    assert np.all(np.isfinite(spectrogram))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"sample_rate": 0}, "sample_rate must be positive"),
        ({"sample_rate": -12000}, "sample_rate must be positive"),
        ({"nperseg": 0}, "nperseg must be positive"),
        ({"nperseg": -1}, "nperseg must be positive"),
        ({"noverlap": -1}, "0 <= noverlap < nperseg"),
        ({"noverlap": 256}, "0 <= noverlap < nperseg"),
        ({"nperseg": 128, "noverlap": 128}, "0 <= noverlap < nperseg"),
    ],
)
def test_compute_rejects_invalid_parameters(
    arguments: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_log_power_spectrogram(np.ones(2048), **arguments)


def test_resize_spectrogram_returns_exact_output_shape() -> None:
    spectrogram = np.arange(15, dtype=np.float64).reshape(3, 5)

    resized = resize_spectrogram(spectrogram)

    assert resized.shape == (224, 224)
    assert resized.dtype == np.float32
    assert np.all(np.isfinite(resized))


def test_standardize_spectrogram_produces_expected_values() -> None:
    spectrogram = np.array([[1.0, 3.0], [5.0, 7.0]])

    standardized = standardize_spectrogram(
        spectrogram,
        mean=4.0,
        std=2.0,
        eps=0.0 + 1e-8,
    )

    expected = (spectrogram - 4.0) / (2.0 + 1e-8)
    np.testing.assert_allclose(standardized, expected.astype(np.float32))
    assert standardized.dtype == np.float32


def test_functions_do_not_mutate_inputs() -> None:
    signal = np.linspace(-1.0, 1.0, 2048)
    signal_original = signal.copy()
    spectrogram = np.arange(12, dtype=np.float64).reshape(3, 4)
    spectrogram_original = spectrogram.copy()

    compute_log_power_spectrogram(signal)
    resize_spectrogram(spectrogram)
    standardize_spectrogram(spectrogram, mean=5.5, std=2.0)

    np.testing.assert_array_equal(signal, signal_original)
    np.testing.assert_array_equal(spectrogram, spectrogram_original)


@pytest.mark.parametrize(
    "spectrogram",
    [
        np.ones(4),
        np.array([["invalid"]]),
        np.array([[1.0, np.inf]]),
    ],
)
def test_resize_rejects_invalid_spectrograms(
    spectrogram: np.ndarray,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        resize_spectrogram(spectrogram)
