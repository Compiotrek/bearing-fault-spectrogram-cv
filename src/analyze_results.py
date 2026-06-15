"""Generate robustness analysis figures from evaluation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

VARIANT_PATTERN = re.compile(r"^(noisy|denoised)_(-?\d+)db$")


def _variant_sort_key(variant: str) -> tuple[int, int, str]:
    if variant == "clean":
        return (0, 0, variant)
    match = VARIANT_PATTERN.match(variant)
    if match is None:
        return (3, 0, variant)
    condition, snr_text = match.groups()
    condition_order = 1 if condition == "noisy" else 2
    return (condition_order, -int(snr_text), variant)


def _snr_pairs(
    variants: dict[str, dict[str, Any]],
) -> dict[int, tuple[str, str]]:
    noisy: dict[int, str] = {}
    denoised: dict[int, str] = {}
    for variant in variants:
        match = VARIANT_PATTERN.match(variant)
        if match is None:
            continue
        condition, snr_text = match.groups()
        snr = int(snr_text)
        if condition == "noisy":
            noisy[snr] = variant
        else:
            denoised[snr] = variant
    return {
        snr: (noisy[snr], denoised[snr])
        for snr in sorted(noisy.keys() & denoised.keys(), reverse=True)
    }


def _save_metric_bar_chart(
    variants: list[str],
    values: list[float],
    metric_name: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bars = axis.bar(variants, values, color="#2563eb")
    axis.set(
        title=f"{metric_name.replace('_', ' ').title()} by Variant",
        xlabel="Variant",
        ylabel=metric_name.replace("_", " ").title(),
        ylim=(0.0, 1.05),
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.02, 1.02),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_recall_heatmap(
    variants: list[str],
    class_names: list[str],
    variant_metrics: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    recall_matrix = np.asarray(
        [
            [
                variant_metrics[variant]["per_class_recall"][class_name]
                for class_name in class_names
            ]
            for variant in variants
        ],
        dtype=np.float64,
    )
    figure, axis = plt.subplots(
        figsize=(9, max(5, len(variants) * 0.65)),
        constrained_layout=True,
    )
    image = axis.imshow(recall_matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    axis.set(
        title="Per-Class Recall by Variant",
        xlabel="Class",
        ylabel="Variant",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(variants)),
        xticklabels=class_names,
        yticklabels=variants,
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    for row in range(recall_matrix.shape[0]):
        for column in range(recall_matrix.shape[1]):
            value = recall_matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, label="Recall")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _effect_description(delta: float) -> str:
    if delta > 1e-12:
        return "improves"
    if delta < -1e-12:
        return "hurts"
    return "does not change"


def _write_key_findings(
    output_path: Path,
    variants: list[str],
    variant_metrics: dict[str, dict[str, Any]],
    class_names: list[str],
    pairs: dict[int, tuple[str, str]],
) -> None:
    best_variant = max(variants, key=lambda name: variant_metrics[name]["accuracy"])
    worst_variant = min(variants, key=lambda name: variant_metrics[name]["accuracy"])
    clean_metrics = variant_metrics.get("clean")

    noisy_variants = [variant for variant in variants if variant.startswith("noisy_")]
    noisy_trend = ", ".join(
        f"{variant.removeprefix('noisy_')}: {variant_metrics[variant]['accuracy']:.3f}"
        for variant in noisy_variants
    )

    lines = [
        "# Key Findings",
        "",
        f"- **Best variant by accuracy:** `{best_variant}` "
        f"({variant_metrics[best_variant]['accuracy']:.3f}).",
        f"- **Worst variant by accuracy:** `{worst_variant}` "
        f"({variant_metrics[worst_variant]['accuracy']:.3f}).",
    ]
    if clean_metrics is not None:
        lines.append(
            "- **Clean performance:** "
            f"accuracy {clean_metrics['accuracy']:.3f}, "
            f"macro F1 {clean_metrics['macro_f1']:.3f}."
        )
    if noisy_trend:
        lines.append(
            "- **Noisy performance trend:** "
            f"{noisy_trend}. Lower SNR indicates stronger noise."
        )

    lines.extend(["", "## Denoising Effect", ""])
    if pairs:
        for snr, (noisy_variant, denoised_variant) in pairs.items():
            noisy_accuracy = variant_metrics[noisy_variant]["accuracy"]
            denoised_accuracy = variant_metrics[denoised_variant]["accuracy"]
            delta = denoised_accuracy - noisy_accuracy
            lines.append(
                f"- **{snr} dB:** denoising {_effect_description(delta)} accuracy "
                f"by {delta:+.3f} ({noisy_accuracy:.3f} to "
                f"{denoised_accuracy:.3f})."
            )
    else:
        lines.append("- No matching noisy/denoised SNR pairs were available.")

    lines.extend(["", "## Weakest Recall", ""])
    weakest_overall: tuple[float, str, str] | None = None
    for class_name in class_names:
        weakest_variant = min(
            variants,
            key=lambda name: variant_metrics[name]["per_class_recall"][class_name],
        )
        weakest_recall = variant_metrics[weakest_variant]["per_class_recall"][
            class_name
        ]
        lines.append(f"- `{class_name}`: {weakest_recall:.3f} on `{weakest_variant}`.")
        candidate = (weakest_recall, class_name, weakest_variant)
        if weakest_overall is None or candidate < weakest_overall:
            weakest_overall = candidate
    if weakest_overall is not None:
        recall, class_name, variant = weakest_overall
        lines.append(
            f"- **Overall weakest:** `{class_name}` on `{variant}` "
            f"with recall {recall:.3f}."
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_results(
    metrics_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create robustness figures and findings from evaluation metrics."""
    metrics_file_path = Path(metrics_path)
    with metrics_file_path.open(encoding="utf-8") as metrics_file:
        evaluation = json.load(metrics_file)
    variant_metrics = evaluation.get("variants")
    if not isinstance(variant_metrics, dict) or not variant_metrics:
        raise ValueError("evaluation metrics must contain a non-empty 'variants' map")

    variants = sorted(variant_metrics, key=_variant_sort_key)
    first_recalls = variant_metrics[variants[0]].get("per_class_recall")
    if not isinstance(first_recalls, dict) or not first_recalls:
        raise ValueError("variant metrics must contain per_class_recall values")
    class_names = list(first_recalls)
    for variant in variants:
        metrics = variant_metrics[variant]
        for key in ("accuracy", "macro_f1", "per_class_recall"):
            if key not in metrics:
                raise ValueError(f"variant {variant!r} is missing metric {key!r}")
        if set(metrics["per_class_recall"]) != set(class_names):
            raise ValueError("per-class recall labels must match across variants")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _save_metric_bar_chart(
        variants,
        [variant_metrics[variant]["accuracy"] for variant in variants],
        "accuracy",
        output_path / "accuracy_by_variant.png",
    )
    _save_metric_bar_chart(
        variants,
        [variant_metrics[variant]["macro_f1"] for variant in variants],
        "macro_f1",
        output_path / "macro_f1_by_variant.png",
    )
    _save_recall_heatmap(
        variants,
        class_names,
        variant_metrics,
        output_path / "per_class_recall_heatmap.png",
    )

    pairs = _snr_pairs(variant_metrics)
    pair_deltas = {
        denoised_variant: (
            variant_metrics[denoised_variant]["accuracy"]
            - variant_metrics[noisy_variant]["accuracy"]
        )
        for noisy_variant, denoised_variant in pairs.values()
    }
    recall_fields = [f"recall_{class_name}" for class_name in class_names]
    summary_path = output_path / "robustness_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "variant",
                "accuracy",
                "macro_f1",
                "denoising_accuracy_delta",
                *recall_fields,
            ],
        )
        writer.writeheader()
        for variant in variants:
            metrics = variant_metrics[variant]
            row = {
                "variant": variant,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "denoising_accuracy_delta": pair_deltas.get(variant, ""),
            }
            row.update(
                {
                    f"recall_{class_name}": metrics["per_class_recall"][class_name]
                    for class_name in class_names
                }
            )
            writer.writerow(row)

    _write_key_findings(
        output_path / "key_findings.md",
        variants,
        variant_metrics,
        class_names,
        pairs,
    )
    return {
        "variants": variants,
        "denoising_accuracy_deltas": pair_deltas,
        "output_dir": str(output_path),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the robustness-analysis CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(arguments: list[str] | None = None) -> None:
    """Run robustness analysis from command-line arguments."""
    args = build_argument_parser().parse_args(arguments)
    analyze_results(args.metrics, args.output_dir)


if __name__ == "__main__":
    main()
