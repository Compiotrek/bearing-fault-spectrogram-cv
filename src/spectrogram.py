"""Spectrogram generation and preprocessing utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom
from scipy.signal import stft


def _validate_numeric_array(
    array: np.ndarray,
    *,
    name: str,
    ndim: int,
) -> NDArray[np.number]:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real values")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def compute_log_power_spectrogram(
    signal: np.ndarray,
    sample_rate: int = 12000,
    nperseg: int = 256,
    noverlap: int = 128,
    eps: float = 1e-10,
) -> NDArray[np.float32]:
    """Compute a finite log-power STFT spectrogram."""
    validated_signal = _validate_numeric_array(signal, name="signal", ndim=1)

    if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(
        sample_rate, (int, np.integer)
    ):
        raise TypeError("sample_rate must be an integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if isinstance(nperseg, (bool, np.bool_)) or not isinstance(
        nperseg, (int, np.integer)
    ):
        raise TypeError("nperseg must be an integer")
    if nperseg <= 0:
        raise ValueError("nperseg must be positive")
    if isinstance(noverlap, (bool, np.bool_)) or not isinstance(
        noverlap, (int, np.integer)
    ):
        raise TypeError("noverlap must be an integer")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap must satisfy 0 <= noverlap < nperseg")
    if isinstance(eps, (bool, np.bool_)) or not isinstance(
        eps, (int, float, np.integer, np.floating)
    ):
        raise TypeError("eps must be a number")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")

    _, _, coefficients = stft(
        validated_signal.astype(np.float64, copy=False),
        fs=int(sample_rate),
        nperseg=int(nperseg),
        noverlap=int(noverlap),
    )
    power = np.square(np.abs(coefficients))
    log_power = np.log10(power + float(eps))
    if not np.all(np.isfinite(log_power)):
        raise ValueError("computed spectrogram contains non-finite values")
    return np.asarray(log_power, dtype=np.float32)


def resize_spectrogram(
    spectrogram: np.ndarray,
    output_shape: tuple[int, int] = (224, 224),
) -> NDArray[np.float32]:
    """Resize a two-dimensional spectrogram."""
    validated = _validate_numeric_array(
        spectrogram,
        name="spectrogram",
        ndim=2,
    )
    if (
        not isinstance(output_shape, tuple)
        or len(output_shape) != 2
        or any(
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            for size in output_shape
        )
    ):
        raise TypeError("output_shape must be a tuple of two integers")
    if any(size <= 0 for size in output_shape):
        raise ValueError("output_shape dimensions must be positive")

    zoom_factors = tuple(
        target / current
        for target, current in zip(output_shape, validated.shape, strict=True)
    )
    resized = zoom(
        validated.astype(np.float64, copy=False),
        zoom=zoom_factors,
        order=1,
    )
    if resized.shape != output_shape:
        raise RuntimeError(
            f"resized spectrogram has shape {resized.shape}, expected {output_shape}"
        )
    if not np.all(np.isfinite(resized)):
        raise ValueError("resized spectrogram contains non-finite values")
    return np.asarray(resized, dtype=np.float32)


def standardize_spectrogram(
    spectrogram: np.ndarray,
    mean: float,
    std: float,
    eps: float = 1e-8,
) -> NDArray[np.float32]:
    """Standardize a spectrogram using externally computed statistics."""
    validated = _validate_numeric_array(
        spectrogram,
        name="spectrogram",
        ndim=2,
    )
    for name, value in (("mean", mean), ("std", std), ("eps", eps)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a number")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if std < 0:
        raise ValueError("std must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")

    standardized = (validated.astype(np.float64, copy=False) - float(mean)) / (
        float(std) + float(eps)
    )
    if not np.all(np.isfinite(standardized)):
        raise ValueError("standardized spectrogram contains non-finite values")
    return np.asarray(standardized, dtype=np.float32)
