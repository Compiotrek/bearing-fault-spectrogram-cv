import numpy as np
import pytest

from src.noise import add_gaussian_noise, measure_snr_db


def test_adding_noise_preserves_shape_and_returns_float32() -> None:
    signal = np.linspace(-1.0, 1.0, 2048, dtype=np.float64)

    noisy = add_gaussian_noise(signal, snr_db=10.0, seed=1)

    assert noisy.shape == signal.shape
    assert noisy.dtype == np.float32


def test_adding_noise_is_reproducible_with_seed() -> None:
    signal = np.linspace(-1.0, 1.0, 2048)

    first = add_gaussian_noise(signal, snr_db=5.0, seed=42)
    second = add_gaussian_noise(signal, snr_db=5.0, seed=42)

    np.testing.assert_array_equal(first, second)


def test_measured_snr_is_close_to_requested_snr() -> None:
    time = np.arange(12000) / 12000
    clean = np.sin(2 * np.pi * 1000 * time).astype(np.float32)

    noisy = add_gaussian_noise(clean, snr_db=5.0, seed=7)

    assert measure_snr_db(clean, noisy) == pytest.approx(5.0, abs=0.5)


def test_zero_power_signal_raises_value_error() -> None:
    with pytest.raises(ValueError, match="zero-power"):
        add_gaussian_noise(np.zeros(128), snr_db=10.0)


def test_measure_snr_db_for_known_arrays() -> None:
    clean = np.ones(4)
    noisy = clean + np.full(4, 0.5)

    assert measure_snr_db(clean, noisy) == pytest.approx(6.0206, abs=1e-4)


def test_measure_snr_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        measure_snr_db(np.ones(4), np.ones(5))


@pytest.mark.parametrize(
    "signal",
    [
        np.ones((2, 2)),
        np.array(["invalid"]),
        np.array([1.0, np.nan]),
    ],
)
def test_add_noise_rejects_invalid_signals(signal: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        add_gaussian_noise(signal, snr_db=10.0)


def test_measure_snr_rejects_zero_noise_power() -> None:
    signal = np.arange(4, dtype=np.float32)

    with pytest.raises(ValueError, match="noise power is zero"):
        measure_snr_db(signal, signal.copy())
