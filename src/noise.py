"""Noise injection and signal-to-noise ratio utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_signal(signal: np.ndarray, name: str) -> NDArray[np.number]:
    if not isinstance(signal, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if signal.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(signal.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if np.issubdtype(signal.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real values")
    if signal.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{name} must contain only finite values")
    return signal


def add_gaussian_noise(
    signal: np.ndarray,
    snr_db: float,
    seed: int | None = None,
) -> NDArray[np.float32]:
    """Add Gaussian noise scaled to the requested signal-to-noise ratio."""
    validated_signal = _validate_signal(signal, "signal")
    if not isinstance(snr_db, (int, float, np.integer, np.floating)):
        raise TypeError("snr_db must be a number")
    if isinstance(snr_db, (bool, np.bool_)) or not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")

    signal_float = validated_signal.astype(np.float64, copy=False)
    signal_power = float(np.mean(np.square(signal_float)))
    if signal_power == 0.0:
        raise ValueError("cannot add noise to a zero-power signal")

    snr_linear = float(np.power(10.0, float(snr_db) / 10.0))
    noise_power = signal_power / snr_linear
    if not np.isfinite(noise_power) or noise_power <= 0.0:
        raise ValueError("snr_db produces an invalid noise power")

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=validated_signal.shape)
    generated_power = float(np.mean(np.square(noise)))
    noise *= np.sqrt(noise_power / generated_power)

    return (signal_float + noise).astype(np.float32)


def measure_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Measure the SNR between a clean signal and its noisy counterpart."""
    clean_signal = _validate_signal(clean, "clean")
    noisy_signal = _validate_signal(noisy, "noisy")
    if clean_signal.shape != noisy_signal.shape:
        raise ValueError("clean and noisy signals must have the same shape")

    clean_float = clean_signal.astype(np.float64, copy=False)
    noisy_float = noisy_signal.astype(np.float64, copy=False)
    signal_power = float(np.mean(np.square(clean_float)))
    noise_power = float(np.mean(np.square(noisy_float - clean_float)))
    if noise_power == 0.0:
        raise ValueError("cannot measure SNR when noise power is zero")

    with np.errstate(divide="ignore"):
        return float(10.0 * np.log10(signal_power / noise_power))
