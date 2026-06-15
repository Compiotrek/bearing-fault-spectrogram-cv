import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset import MANIFEST_FIELDS, ProcessedSample, class_mapping
from src.evaluate import evaluate_checkpoint, main
from src.model import SmallSpectrogramCNN


def make_evaluation_fixture(tmp_path: Path) -> tuple[Path, Path]:
    samples: list[ProcessedSample] = []
    for variant_index, variant in enumerate(("clean", "noisy_5db")):
        for label_id, label in enumerate(("normal", "ball")):
            spectrogram_path = tmp_path / f"{variant}_{label}.npy"
            np.save(
                spectrogram_path,
                np.full(
                    (16, 16),
                    fill_value=float(label_id + variant_index),
                    dtype=np.float32,
                ),
                allow_pickle=False,
            )
            samples.append(
                ProcessedSample(
                    sample_id=f"recording_start_00000000_{variant}_{label}",
                    recording_path=tmp_path / f"{label}.mat",
                    spectrogram_path=spectrogram_path,
                    label=label,
                    label_id=label_id,
                    load=3,
                    split="test",
                    variant=variant,
                    window_start=0,
                    signal_key="X001_DE_time",
                    sample_rate=12000,
                )
            )

    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    **sample.__dict__,
                    "recording_path": str(sample.recording_path),
                    "spectrogram_path": str(sample.spectrogram_path),
                }
            )

    model = SmallSpectrogramCNN(num_classes=4)
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "small_cnn",
            "num_classes": 4,
            "class_mapping": class_mapping(),
            "variant": "clean",
            "preprocessing_config": {
                "input_channels": 1,
                "normalization": None,
            },
            "best_val_accuracy": 0.5,
            "best_epoch": 1,
            "seed": 42,
        },
        checkpoint_path,
    )
    return manifest_path, checkpoint_path


def test_evaluation_cli_creates_metrics_summary_and_figures(
    tmp_path: Path,
) -> None:
    manifest_path, checkpoint_path = make_evaluation_fixture(tmp_path)
    output_dir = tmp_path / "reports"

    main(
        [
            "--manifest",
            str(manifest_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--split",
            "test",
            "--variants",
            "clean",
            "noisy_5db",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    )

    metrics_path = output_dir / "evaluation_metrics.json"
    summary_path = output_dir / "summary.csv"
    assert metrics_path.is_file()
    assert summary_path.is_file()
    assert (output_dir / "figures/confusion_matrix_clean.png").is_file()
    assert (output_dir / "figures/confusion_matrix_noisy_5db.png").is_file()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["model_name"] == "small_cnn"
    for variant in ("clean", "noisy_5db"):
        assert {
            "accuracy",
            "macro_f1",
            "per_class_recall",
            "confusion_matrix",
        } <= metrics["variants"][variant].keys()


def test_evaluation_rejects_missing_variant(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = make_evaluation_fixture(tmp_path)

    with pytest.raises(ValueError, match="no samples found"):
        evaluate_checkpoint(
            manifest=manifest_path,
            checkpoint=checkpoint_path,
            output_dir=tmp_path / "reports",
            split="test",
            variants=("clean", "noisy_0db"),
            batch_size=2,
            device="cpu",
        )


def test_evaluation_rebuilds_model_from_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    manifest_path, checkpoint_path = make_evaluation_fixture(tmp_path)

    results = evaluate_checkpoint(
        manifest=manifest_path,
        checkpoint=checkpoint_path,
        output_dir=tmp_path / "reports",
        split="test",
        variants=("clean",),
        batch_size=2,
        device="cpu",
    )

    assert results["model_name"] == "small_cnn"
    assert results["num_classes"] == 4
    assert results["class_mapping"] == class_mapping()


def test_evaluation_rejects_unknown_checkpoint_model(tmp_path: Path) -> None:
    manifest_path, checkpoint_path = make_evaluation_fixture(tmp_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint["model_name"] = "unknown"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="unsupported model_name"):
        evaluate_checkpoint(
            manifest=manifest_path,
            checkpoint=checkpoint_path,
            output_dir=tmp_path / "reports",
            variants=("clean",),
            device="cpu",
        )
