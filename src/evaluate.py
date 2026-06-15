"""Command-line robustness evaluation for trained spectrogram classifiers."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from src.dataset import ProcessedSample, SpectrogramDataset, load_manifest
from src.model import build_model

DEFAULT_VARIANTS = (
    "clean",
    "noisy_10db",
    "noisy_5db",
    "noisy_0db",
    "denoised_10db",
    "denoised_5db",
    "denoised_0db",
)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return torch.device("cuda")
    raise ValueError("device must be 'auto', 'cpu', or 'cuda'")


def _sample_key(sample: ProcessedSample) -> tuple[str, int, str, int]:
    return (
        sample.recording_path.as_posix(),
        sample.window_start,
        sample.label,
        sample.load,
    )


def _aligned_variant_samples(
    samples: list[ProcessedSample],
    split: str,
    variants: Sequence[str],
) -> dict[str, list[ProcessedSample]]:
    samples_by_variant: dict[str, dict[tuple[str, int, str, int], ProcessedSample]] = {}
    for variant in variants:
        variant_samples = [
            sample
            for sample in samples
            if sample.split == split and sample.variant == variant
        ]
        if not variant_samples:
            raise ValueError(
                f"no samples found for split {split!r} and variant {variant!r}"
            )
        samples_by_variant[variant] = {
            _sample_key(sample): sample for sample in variant_samples
        }

    common_keys = set.intersection(
        *(set(variant_samples) for variant_samples in samples_by_variant.values())
    )
    if not common_keys:
        raise ValueError(f"selected variants have no common samples in split {split!r}")
    ordered_keys = sorted(common_keys)
    return {
        variant: [variant_samples[key] for key in ordered_keys]
        for variant, variant_samples in samples_by_variant.items()
    }


def _compute_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[int(target), int(prediction)] += 1

    recalls: dict[str, float] = {}
    f1_scores: list[float] = []
    for class_id, class_name in enumerate(class_names):
        true_positive = float(confusion[class_id, class_id])
        false_negative = float(confusion[class_id, :].sum() - true_positive)
        false_positive = float(confusion[:, class_id].sum() - true_positive)
        recall_denominator = true_positive + false_negative
        precision_denominator = true_positive + false_positive
        recall = true_positive / recall_denominator if recall_denominator > 0 else 0.0
        precision = (
            true_positive / precision_denominator if precision_denominator > 0 else 0.0
        )
        f1_denominator = precision + recall
        f1 = 2.0 * precision * recall / f1_denominator if f1_denominator > 0 else 0.0
        recalls[class_name] = recall
        f1_scores.append(f1)

    return {
        "accuracy": float(np.mean(targets == predictions)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class_recall": recalls,
        "confusion_matrix": confusion.tolist(),
        "sample_count": int(targets.size),
    }


def _save_confusion_matrix(
    confusion_matrix: list[list[int]],
    class_names: list[str],
    variant: str,
    output_path: Path,
) -> None:
    matrix = np.asarray(confusion_matrix, dtype=np.int64)
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        title=f"Confusion Matrix: {variant}",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.colorbar(image, ax=axis)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def evaluate_checkpoint(
    manifest: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    split: str = "test",
    variants: Sequence[str] = DEFAULT_VARIANTS,
    batch_size: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate one checkpoint across aligned spectrogram variants."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not variants:
        raise ValueError("at least one variant must be selected")

    resolved_device = _resolve_device(device)
    checkpoint_data = torch.load(
        Path(checkpoint),
        map_location=resolved_device,
        weights_only=False,
    )
    model_name = checkpoint_data["model_name"]
    num_classes = int(checkpoint_data["num_classes"])
    model = build_model(
        model_name,
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(resolved_device)
    model.eval()

    mapping = checkpoint_data["class_mapping"]
    class_names = [
        class_name
        for class_name, _ in sorted(mapping.items(), key=lambda item: item[1])
    ]
    if len(class_names) != num_classes:
        raise ValueError("checkpoint class_mapping does not match num_classes")

    samples = load_manifest(manifest)
    aligned_samples = _aligned_variant_samples(samples, split, variants)
    output_path = Path(output_dir)
    figures_path = output_path / "figures"
    output_path.mkdir(parents=True, exist_ok=True)

    variant_metrics: dict[str, dict[str, Any]] = {}
    for variant in variants:
        dataset = SpectrogramDataset(aligned_samples[variant])
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        targets: list[int] = []
        predictions: list[int] = []
        with torch.no_grad():
            for images, labels in data_loader:
                logits = model(images.to(resolved_device))
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
                targets.extend(labels.tolist())

        metrics = _compute_metrics(
            np.asarray(targets, dtype=np.int64),
            np.asarray(predictions, dtype=np.int64),
            class_names,
        )
        variant_metrics[variant] = metrics
        _save_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            variant,
            figures_path / f"confusion_matrix_{variant}.png",
        )

    results: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "model_name": model_name,
        "num_classes": num_classes,
        "class_mapping": mapping,
        "split": split,
        "device": str(resolved_device),
        "variants": variant_metrics,
    }
    with (output_path / "evaluation_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(results, metrics_file, indent=2)
        metrics_file.write("\n")

    recall_fields = [f"recall_{class_name}" for class_name in class_names]
    with (output_path / "summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "variant",
                "sample_count",
                "accuracy",
                "macro_f1",
                *recall_fields,
            ],
        )
        writer.writeheader()
        for variant in variants:
            metrics = variant_metrics[variant]
            row = {
                "variant": variant,
                "sample_count": metrics["sample_count"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
            row.update(
                {
                    f"recall_{class_name}": metrics["per_class_recall"][class_name]
                    for class_name in class_names
                }
            )
            writer.writerow(row)
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def main(arguments: list[str] | None = None) -> None:
    """Run robustness evaluation from command-line arguments."""
    args = build_argument_parser().parse_args(arguments)
    evaluate_checkpoint(
        manifest=args.manifest,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        variants=args.variants,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
