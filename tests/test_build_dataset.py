from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from src.build_dataset import main


def make_raw_fixture(tmp_path: Path) -> Path:
    raw_root = tmp_path / "data/raw/cwru"
    recording_path = raw_root / "load_0/normal/Normal_0.mat"
    recording_path.parent.mkdir(parents=True)
    time = np.arange(512) / 12000
    signal = np.sin(2 * np.pi * 1000 * time) + 0.05
    savemat(recording_path, {"X097_DE_time": signal.reshape(-1, 1)})
    return raw_root


def test_build_dataset_cli_creates_manifest_and_prints_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_root = make_raw_fixture(tmp_path)
    output_root = tmp_path / "data/processed"

    main(
        [
            "--raw-root",
            str(raw_root),
            "--output-root",
            str(output_root),
            "--window-size",
            "512",
            "--stride",
            "512",
            "--sample-rate",
            "12000",
            "--snr-levels",
            "5",
            "--denoise-lowcut",
            "100",
            "--denoise-highcut",
            "5000",
            "--spectrogram-height",
            "8",
            "--spectrogram-width",
            "10",
            "--seed",
            "42",
        ]
    )

    manifest_path = output_root / "manifest.csv"
    output = capsys.readouterr().out
    assert manifest_path.is_file()
    assert "Created 3 processed samples." in output
    assert f"Manifest: {manifest_path}" in output
    assert "Counts by split:" in output
    assert "train: 3" in output
    assert "Counts by label:" in output
    assert "normal: 3" in output
    assert "Counts by variant:" in output
    assert "clean: 1" in output
    assert "noisy_5db: 1" in output
    assert "denoised_5db: 1" in output


def test_build_dataset_cli_rejects_invalid_raw_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="raw dataset root does not exist"):
        main(
            [
                "--raw-root",
                str(missing_root),
                "--output-root",
                str(tmp_path / "processed"),
            ]
        )
