"""Command-line entry point for building the processed CWRU dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.dataset import ProcessedSample, build_processed_dataset


def _print_counts(
    heading: str,
    samples: list[ProcessedSample],
    attribute: str,
) -> None:
    counts = Counter(getattr(sample, attribute) for sample in samples)
    print(f"{heading}:")
    for value, count in sorted(counts.items()):
        print(f"  {value}: {count}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the processed-dataset CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--sample-rate", type=int, default=12000)
    parser.add_argument(
        "--snr-levels",
        type=int,
        nargs="+",
        default=[10, 5, 0],
    )
    parser.add_argument("--denoise-lowcut", type=float, default=100.0)
    parser.add_argument("--denoise-highcut", type=float, default=5000.0)
    parser.add_argument("--spectrogram-height", type=int, default=224)
    parser.add_argument("--spectrogram-width", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(arguments: list[str] | None = None) -> None:
    """Build the processed dataset from command-line arguments."""
    args = build_argument_parser().parse_args(arguments)
    if not args.raw_root.is_dir():
        raise FileNotFoundError(
            f"raw dataset root does not exist or is not a directory: {args.raw_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    samples = build_processed_dataset(
        raw_root=args.raw_root,
        output_root=args.output_root,
        window_size=args.window_size,
        stride=args.stride,
        sample_rate=args.sample_rate,
        snr_levels=tuple(args.snr_levels),
        denoise_lowcut=args.denoise_lowcut,
        denoise_highcut=args.denoise_highcut,
        spectrogram_shape=(
            args.spectrogram_height,
            args.spectrogram_width,
        ),
        seed=args.seed,
    )

    manifest_path = args.output_root / "manifest.csv"
    print(f"Created {len(samples)} processed samples.")
    print(f"Manifest: {manifest_path}")
    _print_counts("Counts by split", samples, "split")
    _print_counts("Counts by label", samples, "label")
    _print_counts("Counts by variant", samples, "variant")


if __name__ == "__main__":
    main()
