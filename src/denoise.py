"""Simple denoising filters for vibration signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt


def butterworth_bandpass_filter(
    signal: np.ndarray,
    sample_rate: int = 12000,
    lowcut: float = 100.0,
    highcut: float = 5000.0,
    order: int = 4,
) -> NDArray[np.float32]:
    """Apply a zero-phase Butterworth band-pass filter to a signal."""
    if not isinstance(signal, np.ndarray):
        raise TypeError("signal must be a numpy.ndarray")
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not np.issubdtype(signal.dtype, np.number):
        raise TypeError("signal must contain numeric values")
    if np.issubdtype(signal.dtype, np.complexfloating):
        raise ValueError("signal must contain real values")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal must contain only finite values")

    if isinstance(sample_rate, (bool, np.bool_)) or not isinstance(
        sample_rate, (int, np.integer)
    ):
        raise TypeError("sample_rate must be an integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if isinstance(order, (bool, np.bool_)) or not isinstance(
        order, (int, np.integer)
    ):
        raise TypeError("order must be an integer")
    if order <= 0:
        raise ValueError("order must be positive")

    for name, value in (("lowcut", lowcut), ("highcut", highcut)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a number")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if lowcut <= 0:
        raise ValueError("lowcut must be positive")
    if highcut >= sample_rate / 2:
        raise ValueError("highcut must be below the Nyquist frequency")
    if lowcut >= highcut:
        raise ValueError("lowcut must be lower than highcut")

    coefficients = butter(
        order,
        [float(lowcut), float(highcut)],
        btype="bandpass",
        fs=int(sample_rate),
    )
    filtered = filtfilt(*coefficients, signal.astype(np.float64, copy=False))
    return np.asarray(filtered, dtype=np.float32)
