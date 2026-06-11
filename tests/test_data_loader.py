from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from src.data_loader import (
    RecordingMetadata,
    discover_recordings,
    find_drive_end_key,
    infer_metadata_from_path,
    load_mat_signal,
)


def write_mat(path: Path, values: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, values)
    return path


def test_find_drive_end_key_selects_matching_non_metadata_key() -> None:
    mat_dict = {
        "__header__": "metadata",
        "X001_FE_time": np.array([1.0]),
        "X001_DE_time": np.array([2.0]),
    }

    assert find_drive_end_key(mat_dict) == "X001_DE_time"


def test_provided_signal_key_overrides_auto_detection(tmp_path: Path) -> None:
    path = write_mat(
        tmp_path / "recording.mat",
        {
            "X001_DE_time": np.array([[1.0], [2.0]]),
            "custom_signal": np.array([[3.0], [4.0]]),
        },
    )

    signal, selected_key = load_mat_signal(path, signal_key="custom_signal")

    assert selected_key == "custom_signal"
    np.testing.assert_array_equal(signal, np.array([3.0, 4.0], dtype=np.float32))


def test_missing_drive_end_key_raises_value_error(tmp_path: Path) -> None:
    path = write_mat(tmp_path / "recording.mat", {"X001_FE_time": np.arange(3)})

    with pytest.raises(ValueError, match="DE_time"):
        load_mat_signal(path)


def test_missing_provided_signal_key_raises_clear_error(tmp_path: Path) -> None:
    path = write_mat(tmp_path / "recording.mat", {"X001_DE_time": np.arange(3)})

    with pytest.raises(KeyError, match="missing_signal"):
        load_mat_signal(path, signal_key="missing_signal")


def test_non_numeric_selected_array_raises_type_error(tmp_path: Path) -> None:
    path = write_mat(
        tmp_path / "recording.mat",
        {"X001_DE_time": np.array(["not", "numeric"])},
    )

    with pytest.raises(TypeError, match="numeric"):
        load_mat_signal(path)


def test_loaded_signal_is_flattened_float32(tmp_path: Path) -> None:
    path = write_mat(
        tmp_path / "recording.mat",
        {"X001_DE_time": np.array([[1, 2, 3]], dtype=np.int16)},
    )

    signal, selected_key = load_mat_signal(path)

    assert selected_key == "X001_DE_time"
    assert signal.shape == (3,)
    assert signal.dtype == np.float32
    np.testing.assert_array_equal(signal, np.array([1.0, 2.0, 3.0]))


def test_metadata_inference_from_expected_layout(tmp_path: Path) -> None:
    path = write_mat(
        tmp_path / "data/raw/cwru/load_0/normal/sample.mat",
        {"X001_DE_time": np.arange(3)},
    )

    metadata = infer_metadata_from_path(path)

    assert metadata == RecordingMetadata(
        path=path,
        label="normal",
        load=0,
        signal_key="X001_DE_time",
        sample_rate=12000,
    )


def test_invalid_label_raises_value_error(tmp_path: Path) -> None:
    path = write_mat(
        tmp_path / "data/raw/cwru/load_0/cage/sample.mat",
        {"X001_DE_time": np.arange(3)},
    )

    with pytest.raises(ValueError, match="invalid CWRU label"):
        infer_metadata_from_path(path)


@pytest.mark.parametrize("load_directory", ["load_4", "load_x", "speed_0"])
def test_invalid_load_directory_raises_value_error(
    tmp_path: Path,
    load_directory: str,
) -> None:
    path = write_mat(
        tmp_path / f"data/raw/cwru/{load_directory}/normal/sample.mat",
        {"X001_DE_time": np.arange(3)},
    )

    with pytest.raises(ValueError, match="invalid load directory"):
        infer_metadata_from_path(path)


def test_discover_recordings_returns_deterministic_sorted_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data/raw/cwru"
    paths = [
        write_mat(
            root / "load_1/normal/Normal_1.mat",
            {"X097_DE_time": np.arange(3)},
        ),
        write_mat(
            root / "load_0/normal/Normal_0.mat",
            {"X097_DE_time": np.arange(3)},
        ),
        write_mat(
            root / "load_0/ball/B007_z.mat",
            {"X118_DE_time": np.arange(3)},
        ),
        write_mat(
            root / "load_0/ball/B007_a.mat",
            {"X118_DE_time": np.arange(3)},
        ),
    ]

    recordings = discover_recordings(root)

    assert [recording.path for recording in recordings] == [
        paths[3],
        paths[2],
        paths[1],
        paths[0],
    ]
    assert [
        (recording.load, recording.label, recording.path.name)
        for recording in recordings
    ] == [
        (0, "ball", "B007_a.mat"),
        (0, "ball", "B007_z.mat"),
        (0, "normal", "Normal_0.mat"),
        (1, "normal", "Normal_1.mat"),
    ]
