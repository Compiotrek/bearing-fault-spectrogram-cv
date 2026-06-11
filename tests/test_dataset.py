from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from src.dataset import (
    ProcessedSample,
    build_processed_dataset,
    class_mapping,
    load_manifest,
    split_from_load,
)

EXPECTED_VARIANTS = {
    "clean",
    "noisy_10db",
    "noisy_5db",
    "noisy_0db",
    "denoised_10db",
    "denoised_5db",
    "denoised_0db",
}


def write_recording(
    root: Path,
    *,
    load: int,
    label: str,
    filename: str,
    signal_key: str,
) -> Path:
    path = root / f"load_{load}" / label / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(512) / 12000
    signal = (
        np.sin(2 * np.pi * 1000 * time) + 0.25 * np.sin(2 * np.pi * 2000 * time) + 0.05
    )
    savemat(path, {signal_key: signal.reshape(-1, 1)})
    return path


@pytest.fixture
def synthetic_raw_root(tmp_path: Path) -> Path:
    raw_root = tmp_path / "data/raw/cwru"
    write_recording(
        raw_root,
        load=0,
        label="normal",
        filename="Normal_0.mat",
        signal_key="X097_DE_time",
    )
    write_recording(
        raw_root,
        load=2,
        label="ball",
        filename="B007_2.mat",
        signal_key="X120_DE_time",
    )
    write_recording(
        raw_root,
        load=3,
        label="outer_race",
        filename="OR007@6_3.mat",
        signal_key="X133_DE_time",
    )
    return raw_root


@pytest.mark.parametrize(
    ("load", "expected"),
    [(0, "train"), (1, "train"), (2, "val"), (3, "test")],
)
def test_split_from_load(load: int, expected: str) -> None:
    assert split_from_load(load) == expected


def test_split_from_load_rejects_invalid_load() -> None:
    with pytest.raises(ValueError, match="0, 1, 2, or 3"):
        split_from_load(4)


def test_class_mapping_is_deterministic() -> None:
    assert class_mapping() == {
        "normal": 0,
        "ball": 1,
        "inner_race": 2,
        "outer_race": 3,
    }


def test_build_processed_dataset_creates_variants_manifest_and_arrays(
    synthetic_raw_root: Path,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "processed"

    samples = build_processed_dataset(
        synthetic_raw_root,
        output_root,
        window_size=512,
        stride=512,
        spectrogram_shape=(16, 20),
    )

    assert (output_root / "manifest.csv").is_file()
    assert len(samples) == 3 * len(EXPECTED_VARIANTS)
    assert {sample.variant for sample in samples} == EXPECTED_VARIANTS
    assert {(sample.load, sample.split) for sample in samples} == {
        (0, "train"),
        (2, "val"),
        (3, "test"),
    }
    for sample in samples:
        assert sample.spectrogram_path.is_file()
        spectrogram = np.load(sample.spectrogram_path, allow_pickle=False)
        assert spectrogram.shape == (16, 20)
        assert spectrogram.dtype == np.float32
        assert np.all(np.isfinite(spectrogram))
        assert (
            sample.spectrogram_path.parent
            == output_root / "spectrograms" / sample.split / sample.variant
        )


def test_sample_ids_and_noise_are_deterministic_for_same_seed(
    synthetic_raw_root: Path,
    tmp_path: Path,
) -> None:
    first = build_processed_dataset(
        synthetic_raw_root,
        tmp_path / "processed_first",
        window_size=512,
        stride=512,
        spectrogram_shape=(12, 12),
        seed=123,
    )
    second = build_processed_dataset(
        synthetic_raw_root,
        tmp_path / "processed_second",
        window_size=512,
        stride=512,
        spectrogram_shape=(12, 12),
        seed=123,
    )

    assert [sample.sample_id for sample in first] == [
        sample.sample_id for sample in second
    ]
    for first_sample, second_sample in zip(first, second, strict=True):
        first_array = np.load(first_sample.spectrogram_path, allow_pickle=False)
        second_array = np.load(second_sample.spectrogram_path, allow_pickle=False)
        np.testing.assert_array_equal(first_array, second_array)


def test_load_manifest_round_trips_processed_samples(
    synthetic_raw_root: Path,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "processed"
    built_samples = build_processed_dataset(
        synthetic_raw_root,
        output_root,
        window_size=512,
        stride=512,
        snr_levels=(5,),
        spectrogram_shape=(8, 8),
    )

    loaded_samples = load_manifest(output_root / "manifest.csv")

    assert loaded_samples == built_samples
    assert all(isinstance(sample, ProcessedSample) for sample in loaded_samples)
    assert all(isinstance(sample.recording_path, Path) for sample in loaded_samples)
    assert all(isinstance(sample.spectrogram_path, Path) for sample in loaded_samples)
