import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset import MANIFEST_FIELDS, ProcessedSample
from src.train import main, seed_everything, train_model


def write_manifest(path: Path, samples: list[ProcessedSample]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as manifest_file:
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
    return path


def make_training_fixture(tmp_path: Path) -> Path:
    samples: list[ProcessedSample] = []
    for split in ("train", "val"):
        for index in range(2):
            spectrogram_path = tmp_path / f"{split}_clean_{index}.npy"
            spectrogram = np.full(
                (16, 16),
                fill_value=float(index + (split == "val")),
                dtype=np.float32,
            )
            np.save(spectrogram_path, spectrogram, allow_pickle=False)
            samples.append(
                ProcessedSample(
                    sample_id=f"{split}_clean_{index}",
                    recording_path=tmp_path / f"{split}_{index}.mat",
                    spectrogram_path=spectrogram_path,
                    label="normal" if index == 0 else "ball",
                    label_id=index,
                    load=0 if split == "train" else 2,
                    split=split,
                    variant="clean",
                    window_start=index,
                    signal_key="X001_DE_time",
                    sample_rate=12000,
                )
            )

        samples.append(
            ProcessedSample(
                sample_id=f"{split}_unused_variant",
                recording_path=tmp_path / f"{split}_unused.mat",
                spectrogram_path=tmp_path / "missing_unused_variant.npy",
                label="normal",
                label_id=0,
                load=0 if split == "train" else 2,
                split=split,
                variant="noisy_5db",
                window_start=99,
                signal_key="X001_DE_time",
                sample_rate=12000,
            )
        )
    return write_manifest(tmp_path / "manifest.csv", samples)


def test_training_cli_creates_checkpoint_and_metrics(tmp_path: Path) -> None:
    manifest_path = make_training_fixture(tmp_path)
    output_dir = tmp_path / "models"

    main(
        [
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--model-name",
            "small_cnn",
            "--variant",
            "clean",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.001",
            "--seed",
            "7",
            "--device",
            "cpu",
            "--no-pretrained",
        ]
    )

    checkpoint_path = output_dir / "best_model.pt"
    metrics_path = output_dir / "train_metrics.json"
    assert checkpoint_path.is_file()
    assert metrics_path.is_file()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert {
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
        "history",
        "best_epoch",
        "best_val_accuracy",
    } <= metrics.keys()
    assert len(metrics["history"]) == 1
    assert metrics["history"][0]["epoch"] == 1
    assert metrics["best_epoch"] == 1
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["model_name"] == "small_cnn"
    assert checkpoint["variant"] == "clean"
    assert checkpoint["seed"] == 7
    assert checkpoint["best_epoch"] == 1
    assert checkpoint["best_val_accuracy"] == metrics["best_val_accuracy"]
    assert {
        "model_state_dict",
        "num_classes",
        "class_mapping",
        "preprocessing_config",
    } <= checkpoint.keys()


def test_training_uses_only_selected_variant(tmp_path: Path) -> None:
    manifest_path = make_training_fixture(tmp_path)

    metrics = train_model(
        manifest=manifest_path,
        output_dir=tmp_path / "models",
        model_name="small_cnn",
        variant="clean",
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        seed=11,
        device="cpu",
        pretrained=False,
    )

    assert metrics["variant"] == "clean"


def test_seed_everything_can_be_called_deterministically() -> None:
    seed_everything(42)
    first_numpy = np.random.random()
    first_torch = torch.rand(1)

    seed_everything(42)

    assert np.random.random() == first_numpy
    torch.testing.assert_close(torch.rand(1), first_torch)


def test_invalid_model_name_raises_clear_error(tmp_path: Path) -> None:
    manifest_path = make_training_fixture(tmp_path)

    with pytest.raises(ValueError, match="unsupported model_name"):
        train_model(
            manifest=manifest_path,
            output_dir=tmp_path / "models",
            model_name="invalid",
            variant="clean",
            epochs=1,
            batch_size=2,
            device="cpu",
            pretrained=False,
        )


@pytest.mark.parametrize(
    ("missing_split", "message"),
    [
        ("train", "training dataset is empty"),
        ("val", "validation dataset is empty"),
    ],
)
def test_training_rejects_empty_required_split(
    tmp_path: Path,
    missing_split: str,
    message: str,
) -> None:
    manifest_path = make_training_fixture(tmp_path)
    rows = list(csv.DictReader(manifest_path.read_text(encoding="utf-8").splitlines()))
    remaining_rows = [
        row
        for row in rows
        if row["split"] != missing_split or row["variant"] != "clean"
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(remaining_rows)

    with pytest.raises(ValueError, match=message):
        train_model(
            manifest=manifest_path,
            output_dir=tmp_path / "models",
            model_name="small_cnn",
            variant="clean",
            epochs=1,
            batch_size=2,
            device="cpu",
            pretrained=False,
        )
