"""Grad-CAM explanations for trained spectrogram classifiers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from src.dataset import ProcessedSample, load_manifest
from src.model import ResNet18SpectrogramClassifier, build_model


def _target_layer(model: nn.Module) -> nn.Module:
    if isinstance(model, ResNet18SpectrogramClassifier):
        return model.backbone.layer4[-1]

    convolution_layers = [
        module for module in model.modules() if isinstance(module, nn.Conv2d)
    ]
    if not convolution_layers:
        raise ValueError("model must contain at least one convolutional layer")
    return convolution_layers[-1]


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
) -> np.ndarray:
    """Generate a normalized Grad-CAM heatmap for one input image."""
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError("input_tensor must have shape (1, C, H, W)")
    if not input_tensor.is_floating_point():
        raise TypeError("input_tensor must have a floating-point dtype")
    if not torch.isfinite(input_tensor).all():
        raise ValueError("input_tensor must contain only finite values")

    activations: torch.Tensor | None = None

    def capture_activations(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        nonlocal activations
        activations = output

    handle = _target_layer(model).register_forward_hook(capture_activations)
    grad_input = input_tensor.detach().clone().requires_grad_(True)
    try:
        logits = model(grad_input)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("model output must have shape (1, num_classes)")
        class_id = (
            int(logits.argmax(dim=1).item()) if target_class is None else target_class
        )
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise TypeError("target_class must be an integer or None")
        if not 0 <= class_id < logits.shape[1]:
            raise ValueError(
                f"target_class must be between 0 and {logits.shape[1] - 1}"
            )
        if activations is None:
            raise RuntimeError("target layer did not produce activations")

        gradients = torch.autograd.grad(
            logits[0, class_id],
            activations,
            retain_graph=False,
            create_graph=False,
        )[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(
            heatmap,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        heatmap = heatmap.detach().cpu().numpy().astype(np.float32, copy=False)
    finally:
        handle.remove()

    heatmap -= float(heatmap.min())
    maximum = float(heatmap.max())
    if maximum > 0.0:
        heatmap /= maximum
    return np.clip(heatmap, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_image(array: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")

    normalized = values.astype(np.float32, copy=True)
    normalized -= float(normalized.min())
    value_range = float(normalized.max())
    if value_range > 0.0:
        normalized /= value_range
    return normalized


def overlay_heatmap_on_spectrogram(
    spectrogram: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a normalized Grad-CAM heatmap with a spectrogram."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    normalized_spectrogram = _normalize_image(spectrogram, "spectrogram")
    normalized_heatmap = _normalize_image(heatmap, "heatmap")
    if normalized_spectrogram.shape != normalized_heatmap.shape:
        raise ValueError("spectrogram and heatmap must have matching shapes")

    grayscale = np.repeat(normalized_spectrogram[..., None], 3, axis=2)
    colored_heatmap = plt.get_cmap("jet")(normalized_heatmap)[..., :3]
    overlay = (1.0 - alpha) * grayscale + alpha * colored_heatmap
    return np.clip(overlay, 0.0, 1.0).astype(np.float32)


def save_gradcam_example(
    spectrogram: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Save a spectrogram with its Grad-CAM overlay as a PNG."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_heatmap_on_spectrogram(spectrogram, heatmap)

    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    axis.imshow(overlay, origin="lower", aspect="auto")
    if title is not None:
        axis.set_title(title)
    axis.set_axis_off()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


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


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _selected_samples(
    samples: list[ProcessedSample],
    split: str,
    variant: str,
) -> list[ProcessedSample]:
    selected = [
        sample
        for sample in samples
        if sample.split == split and sample.variant == variant
    ]
    if not selected:
        raise ValueError(
            f"no samples found for split {split!r} and variant {variant!r}"
        )
    return sorted(selected, key=lambda sample: sample.sample_id)


def generate_gradcam_examples(
    manifest: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    split: str = "test",
    variant: str = "clean",
    max_samples: int = 12,
    device: str = "auto",
) -> list[Path]:
    """Generate Grad-CAM PNG examples from a trained checkpoint."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    resolved_device = _resolve_device(device)
    checkpoint_data = torch.load(
        Path(checkpoint),
        map_location=resolved_device,
        weights_only=False,
    )
    model = build_model(
        checkpoint_data["model_name"],
        num_classes=int(checkpoint_data["num_classes"]),
        pretrained=False,
    )
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.to(resolved_device)
    model.eval()

    mapping = checkpoint_data["class_mapping"]
    class_names = {
        int(class_id): class_name for class_name, class_id in mapping.items()
    }
    samples = _selected_samples(load_manifest(manifest), split, variant)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for sample in samples[:max_samples]:
        spectrogram = np.load(sample.spectrogram_path, allow_pickle=False)
        input_tensor = (
            torch.from_numpy(spectrogram.astype(np.float32, copy=False))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(resolved_device)
        )
        with torch.no_grad():
            probabilities = torch.softmax(model(input_tensor), dim=1)
            confidence, predicted_id = probabilities.max(dim=1)
        predicted_class = class_names[int(predicted_id.item())]
        heatmap = generate_gradcam(
            model,
            input_tensor,
            target_class=int(predicted_id.item()),
        )
        title = (
            f"True: {sample.label} | Predicted: {predicted_class} | "
            f"Confidence: {float(confidence.item()):.1%}"
        )
        filename = _safe_filename(
            f"{sample.sample_id}_true-{sample.label}_pred-{predicted_class}"
            f"_conf-{float(confidence.item()):.3f}.png"
        )
        output_path = destination / filename
        save_gradcam_example(spectrogram, heatmap, output_path, title=title)
        saved_paths.append(output_path)
    return saved_paths


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the Grad-CAM CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--variant", default="clean")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Grad-CAM example CLI."""
    args = build_argument_parser().parse_args(argv)
    saved_paths = generate_gradcam_examples(
        manifest=args.manifest,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        variant=args.variant,
        max_samples=args.max_samples,
        device=args.device,
    )
    print(f"Saved {len(saved_paths)} Grad-CAM examples to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
