import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from src.app_utils import (
    build_live_replay_frame,
    filter_samples,
    load_app_manifest,
    load_checkpoint_model,
    load_spectrogram,
    normalize_spectrogram,
    predict_spectrogram,
    replay_elapsed_seconds,
    reveal_spectrogram,
    rolling_spectrogram_window,
    selector_values,
    spectral_activity,
)
from src.dataset import MANIFEST_FIELDS, ProcessedSample, class_mapping
from src.model import SmallSpectrogramCNN


def make_sample(
    path: Path,
    *,
    sample_id: str = "sample",
    split: str = "test",
    variant: str = "clean",
    label: str = "normal",
    label_id: int = 0,
) -> ProcessedSample:
    return ProcessedSample(
        sample_id=sample_id,
        recording_path=Path("data/raw/example.mat"),
        spectrogram_path=path,
        label=label,
        label_id=label_id,
        load=3,
        split=split,
        variant=variant,
        window_start=0,
        signal_key="X_DE_time",
        sample_rate=12000,
    )


def test_filter_samples_and_selector_values_are_deterministic(tmp_path: Path) -> None:
    samples = [
        make_sample(
            tmp_path / "b.npy",
            sample_id="b",
            variant="noisy_5db",
            label="ball",
            label_id=1,
        ),
        make_sample(tmp_path / "a.npy", sample_id="a"),
        make_sample(
            tmp_path / "c.npy",
            sample_id="c",
            split="val",
            label="ball",
            label_id=1,
        ),
    ]

    selected = filter_samples(samples, split="test", variant="clean")

    assert [sample.sample_id for sample in selected] == ["a"]
    assert selector_values(samples, "split") == ["test", "val"]
    assert selector_values(samples, "label") == ["ball", "normal"]


def test_load_app_manifest_resolves_relative_paths(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    manifest_path = project_root / "data" / "processed" / "manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    row = make_sample(Path("data/processed/example.npy"))
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        values = row.__dict__.copy()
        values["recording_path"] = str(row.recording_path)
        values["spectrogram_path"] = str(row.spectrogram_path)
        writer.writerow(values)

    samples = load_app_manifest(manifest_path, project_root=project_root)

    assert samples[0].spectrogram_path == project_root / row.spectrogram_path
    assert samples[0].recording_path == project_root / row.recording_path


def test_load_app_manifest_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest file not found"):
        load_app_manifest(tmp_path / "missing.csv")


def test_load_spectrogram_returns_float32_and_reports_missing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spectrogram.npy"
    np.save(path, np.arange(12, dtype=np.float64).reshape(3, 4))

    spectrogram = load_spectrogram(path)

    assert spectrogram.shape == (3, 4)
    assert spectrogram.dtype == np.float32
    with pytest.raises(FileNotFoundError, match="spectrogram file not found"):
        load_spectrogram(tmp_path / "missing.npy")


def test_load_checkpoint_and_predict_spectrogram(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best_model.pt"
    source_model = SmallSpectrogramCNN(num_classes=4)
    torch.save(
        {
            "model_state_dict": source_model.state_dict(),
            "model_name": "small_cnn",
            "num_classes": 4,
            "class_mapping": class_mapping(),
        },
        checkpoint_path,
    )

    model, mapping, device = load_checkpoint_model(checkpoint_path, device="cpu")
    result = predict_spectrogram(
        model,
        np.random.default_rng(4).normal(size=(32, 32)).astype(np.float32),
        mapping,
        device,
    )

    assert not model.training
    assert result.predicted_class in mapping
    assert 0.0 <= result.confidence <= 1.0
    assert list(result.probabilities) == sorted(
        result.probabilities,
        key=result.probabilities.get,
        reverse=True,
    )
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_load_checkpoint_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint file not found"):
        load_checkpoint_model(tmp_path / "missing.pt", device="cpu")


def test_normalize_spectrogram_handles_constant_values() -> None:
    normalized = normalize_spectrogram(np.full((2, 3), 7.0, dtype=np.float32))

    np.testing.assert_array_equal(normalized, np.zeros((2, 3), dtype=np.float32))


def test_reveal_spectrogram_reveals_columns_from_left_to_right() -> None:
    spectrogram = np.arange(12, dtype=np.float32).reshape(3, 4)

    halfway = reveal_spectrogram(spectrogram, 0.5, hidden_value=-1.0)

    np.testing.assert_array_equal(halfway[:, :2], spectrogram[:, :2])
    np.testing.assert_array_equal(halfway[:, 2:], -1.0)
    np.testing.assert_array_equal(spectrogram, np.arange(12).reshape(3, 4))


def test_rolling_spectrogram_window_moves_history_left() -> None:
    spectrogram = np.arange(48, dtype=np.float32).reshape(6, 8)

    earlier = rolling_spectrogram_window(spectrogram, 3 / 7, viewport_columns=5)
    later = rolling_spectrogram_window(spectrogram, 4 / 7, viewport_columns=5)

    assert earlier.shape == (6, 5)
    np.testing.assert_allclose(earlier[:, 2:5], later[:, 1:4])


def test_build_live_replay_frame_returns_rgb_monitor_view() -> None:
    frame = build_live_replay_frame(
        np.arange(48, dtype=np.float32).reshape(6, 8),
        0.5,
        viewport_columns=8,
    )

    assert frame.shape == (6, 8, 3)
    assert frame.dtype == np.float32
    assert np.all(np.isfinite(frame))
    assert np.all((0.0 <= frame) & (frame <= 1.0))
    expected_scanline = np.tile([0.1, 0.9, 1.0], (4, 1))
    np.testing.assert_allclose(frame[1:-1, 6, :], expected_scanline)


def test_replay_elapsed_seconds_uses_window_duration() -> None:
    elapsed, total = replay_elapsed_seconds(0.5, window_size=2048, sample_rate=12000)

    assert total == pytest.approx(2048 / 12000)
    assert elapsed == pytest.approx(total / 2)


def test_spectral_activity_tracks_current_column() -> None:
    spectrogram = np.zeros((4, 3), dtype=np.float32)
    spectrogram[:, 2] = 10.0

    assert spectral_activity(spectrogram, 0.0) == pytest.approx(0.0)
    assert spectral_activity(spectrogram, 1.0) == pytest.approx(1.0)


@pytest.mark.parametrize("progress", [-0.1, 1.1])
def test_reveal_spectrogram_rejects_invalid_progress(progress: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        reveal_spectrogram(np.zeros((2, 2), dtype=np.float32), progress)
