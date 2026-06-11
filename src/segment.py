"""Utilities for splitting vibration signals into fixed-size windows."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def segment_signal(
    signal: NDArray[np.number],
    window_size: int = 2048,
    stride: int = 1024,
) -> tuple[NDArray[np.number], NDArray[np.intp]]:
    """Split a one-dimensional signal into complete, fixed-size windows.

    Incomplete samples at the end of the signal are discarded. Returned windows
    own their memory, so modifying them never changes the input signal.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("signal must be a numpy.ndarray")
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not np.issubdtype(signal.dtype, np.number):
        raise TypeError("signal must contain numeric values")
    if isinstance(window_size, bool) or not isinstance(window_size, (int, np.integer)):
        raise TypeError("window_size must be an integer")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)):
        raise TypeError("stride must be an integer")
    if stride <= 0:
        raise ValueError("stride must be positive")

    if signal.size < window_size:
        return (
            np.empty((0, window_size), dtype=signal.dtype),
            np.empty((0,), dtype=np.intp),
        )

    starts = np.arange(
        0,
        signal.size - window_size + 1,
        stride,
        dtype=np.intp,
    )
    offsets = np.arange(window_size, dtype=np.intp)
    windows = signal[starts[:, np.newaxis] + offsets].copy()
    return windows, starts
