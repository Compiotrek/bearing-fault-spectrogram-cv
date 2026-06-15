"""Command-line training entry point for spectrogram classifiers."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dataset import SpectrogramDataset, class_mapping, load_manifest
from src.model import build_model


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and available CUDA devices."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def _run_epoch(
    model: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = loss_function(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def train_model(
    manifest: str | Path,
    output_dir: str | Path,
    model_name: str = "resnet18",
    variant: str = "clean",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str = "auto",
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> dict[str, Any]:
    """Train a classifier and persist its best checkpoint and metrics."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    seed_everything(seed)
    resolved_device = _resolve_device(device)
    samples = load_manifest(manifest)
    train_samples = [
        sample
        for sample in samples
        if sample.split == "train" and sample.variant == variant
    ]
    if not train_samples:
        raise ValueError(
            f"training dataset is empty for split 'train' and variant {variant!r}"
        )
    val_samples = [
        sample
        for sample in samples
        if sample.split == "val" and sample.variant == variant
    ]
    if not val_samples:
        raise ValueError(
            f"validation dataset is empty for split 'val' and variant {variant!r}"
        )

    train_dataset = SpectrogramDataset(
        train_samples,
    )
    val_dataset = SpectrogramDataset(
        val_samples,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    labels = class_mapping()
    model = build_model(
        model_name,
        num_classes=len(labels),
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    ).to(resolved_device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "best_model.pt"
    best_val_accuracy = float("-inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model,
            train_loader,
            loss_function,
            resolved_device,
            optimizer,
        )
        val_loss, val_accuracy = _run_epoch(
            model,
            val_loader,
            loss_function,
            resolved_device,
        )
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(epoch_metrics)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "num_classes": len(labels),
                    "class_mapping": labels,
                    "variant": variant,
                    "preprocessing_config": {
                        "input_channels": 1,
                        "normalization": None,
                    },
                    "best_val_accuracy": best_val_accuracy,
                    "best_epoch": best_epoch,
                    "seed": seed,
                },
                checkpoint_path,
            )

    final_epoch = history[-1]
    metrics: dict[str, Any] = {
        "train_loss": final_epoch["train_loss"],
        "train_accuracy": final_epoch["train_accuracy"],
        "val_loss": final_epoch["val_loss"],
        "val_accuracy": final_epoch["val_accuracy"],
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "model_name": model_name,
        "variant": variant,
        "device": str(resolved_device),
        "seed": seed,
        "history": history,
    }
    metrics_path = output_path / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")
    return metrics


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the training CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name",
        default="resnet18",
        choices=("small_cnn", "resnet18"),
    )
    parser.add_argument("--variant", default="clean")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load pretrained ResNet-18 weights.",
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> None:
    """Run training from command-line arguments."""
    args = build_argument_parser().parse_args(arguments)
    train_model(
        manifest=args.manifest,
        output_dir=args.output_dir,
        model_name=args.model_name,
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    )


if __name__ == "__main__":
    main()
