import csv
import json
from pathlib import Path

import pytest

from src.analyze_results import analyze_results, main


def write_metrics(path: Path) -> Path:
    variants = {
        "clean": {
            "accuracy": 0.95,
            "macro_f1": 0.94,
            "per_class_recall": {"normal": 1.0, "ball": 0.9},
            "confusion_matrix": [[10, 0], [1, 9]],
            "sample_count": 20,
        },
        "noisy_5db": {
            "accuracy": 0.60,
            "macro_f1": 0.58,
            "per_class_recall": {"normal": 0.7, "ball": 0.5},
            "confusion_matrix": [[7, 3], [5, 5]],
            "sample_count": 20,
        },
        "denoised_5db": {
            "accuracy": 0.75,
            "macro_f1": 0.73,
            "per_class_recall": {"normal": 0.8, "ball": 0.7},
            "confusion_matrix": [[8, 2], [3, 7]],
            "sample_count": 20,
        },
    }
    path.write_text(
        json.dumps(
            {
                "model_name": "small_cnn",
                "split": "test",
                "variants": variants,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_analysis_cli_creates_summary_findings_and_plots(
    tmp_path: Path,
) -> None:
    metrics_path = write_metrics(tmp_path / "evaluation_metrics.json")
    output_dir = tmp_path / "analysis"

    main(
        [
            "--metrics",
            str(metrics_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert (output_dir / "robustness_summary.csv").is_file()
    assert (output_dir / "key_findings.md").is_file()
    assert (output_dir / "accuracy_by_variant.png").is_file()
    assert (output_dir / "macro_f1_by_variant.png").is_file()
    assert (output_dir / "per_class_recall_heatmap.png").is_file()

    findings = (output_dir / "key_findings.md").read_text(encoding="utf-8")
    assert "Best variant by accuracy" in findings
    assert "denoising improves accuracy by +0.150" in findings
    assert "Overall weakest" in findings


def test_analysis_computes_denoising_delta_correctly(tmp_path: Path) -> None:
    metrics_path = write_metrics(tmp_path / "evaluation_metrics.json")
    output_dir = tmp_path / "analysis"

    results = analyze_results(metrics_path, output_dir)

    assert results["denoising_accuracy_deltas"]["denoised_5db"] == pytest.approx(0.15)
    with (output_dir / "robustness_summary.csv").open(
        encoding="utf-8",
        newline="",
    ) as summary_file:
        rows = {row["variant"]: row for row in csv.DictReader(summary_file)}
    assert float(rows["denoised_5db"]["denoising_accuracy_delta"]) == pytest.approx(
        0.15
    )
