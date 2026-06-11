import numpy as np
import pytest

from src.segment import segment_signal


def test_exact_size_signal_creates_one_window() -> None:
    signal = np.arange(8, dtype=np.float32)

    windows, starts = segment_signal(signal, window_size=8, stride=4)

    np.testing.assert_array_equal(windows, signal[np.newaxis, :])
    np.testing.assert_array_equal(starts, np.array([0]))


def test_overlap_produces_expected_windows_and_start_indices() -> None:
    signal = np.arange(10)

    windows, starts = segment_signal(signal, window_size=4, stride=2)

    np.testing.assert_array_equal(
        windows,
        np.array(
            [
                [0, 1, 2, 3],
                [2, 3, 4, 5],
                [4, 5, 6, 7],
                [6, 7, 8, 9],
            ]
        ),
    )
    np.testing.assert_array_equal(starts, np.array([0, 2, 4, 6]))


def test_trailing_incomplete_samples_are_dropped() -> None:
    signal = np.arange(11)

    windows, starts = segment_signal(signal, window_size=4, stride=4)

    np.testing.assert_array_equal(
        windows,
        np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
    )
    np.testing.assert_array_equal(starts, np.array([0, 4]))


@pytest.mark.parametrize("signal", [np.array([]), np.arange(3)])
def test_signal_without_complete_window_returns_shaped_empty_arrays(
    signal: np.ndarray,
) -> None:
    windows, starts = segment_signal(signal, window_size=4, stride=2)

    assert windows.shape == (0, 4)
    assert starts.shape == (0,)
    assert windows.dtype == signal.dtype
    assert np.issubdtype(starts.dtype, np.integer)


@pytest.mark.parametrize(
    ("signal", "error", "message"),
    [
        (np.zeros((2, 2)), ValueError, "one-dimensional"),
        (np.array(["a", "b"]), TypeError, "numeric"),
        ([1, 2, 3], TypeError, "numpy.ndarray"),
    ],
)
def test_invalid_signal_raises_clear_error(
    signal: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        segment_signal(signal)  # type: ignore[arg-type]


@pytest.mark.parametrize("window_size", [0, -1])
def test_non_positive_window_size_is_rejected(window_size: int) -> None:
    with pytest.raises(ValueError, match="window_size must be positive"):
        segment_signal(np.arange(8), window_size=window_size)


@pytest.mark.parametrize("stride", [0, -1])
def test_non_positive_stride_is_rejected(stride: int) -> None:
    with pytest.raises(ValueError, match="stride must be positive"):
        segment_signal(np.arange(8), stride=stride)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("window_size", 2.5),
        ("window_size", True),
        ("stride", 2.5),
        ("stride", False),
    ],
)
def test_non_integer_window_parameters_are_rejected(
    argument: str,
    value: object,
) -> None:
    with pytest.raises(TypeError, match=f"{argument} must be an integer"):
        segment_signal(np.arange(8), **{argument: value})  # type: ignore[arg-type]


def test_output_preserves_dtype_and_owns_its_memory() -> None:
    signal = np.arange(8, dtype=np.int16)

    windows, _ = segment_signal(signal, window_size=4, stride=2)
    windows[0, 0] = 99

    assert windows.dtype == signal.dtype
    assert windows.flags.owndata
    assert signal[0] == 0
