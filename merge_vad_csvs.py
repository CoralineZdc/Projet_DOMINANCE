from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_required_columns(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError("CSV not found: {}".format(csv_path))
    df = pd.read_csv(csv_path)
    required = ["pixels", "Valence", "Arousal", "Dominance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV is missing required columns {}: {}".format(missing, csv_path))
    return df


def merge_csvs(base_train_csv: Path, extra_csvs: list[Path], output_csv: Path, deduplicate: bool) -> int:
    dataframes = [load_required_columns(base_train_csv)]
    for extra in extra_csvs:
        dataframes.append(load_required_columns(extra))

    merged = pd.concat(dataframes, ignore_index=True)
    if deduplicate:
        merged = merged.drop_duplicates(subset=["pixels", "Valence", "Arousal", "Dominance"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return len(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple VAD CSV files into one training CSV")
    parser.add_argument("--base_train_csv", type=str, default="data/train-20240123-14902.csv")
    parser.add_argument("--extra_csvs", type=str, required=True, help="Comma-separated CSV paths")
    parser.add_argument("--output_csv", type=str, default="data/train-merged-vad-all.csv")
    parser.add_argument("--no_dedup", action="store_true")
    args = parser.parse_args()

    base = Path(args.base_train_csv)
    extras = [Path(x.strip()) for x in args.extra_csvs.split(",") if x.strip()]
    if not extras:
        raise ValueError("Provide at least one extra CSV via --extra_csvs")

    total_rows = merge_csvs(
        base_train_csv=base,
        extra_csvs=extras,
        output_csv=Path(args.output_csv),
        deduplicate=not args.no_dedup,
    )
    print("Wrote {} rows to {}".format(total_rows, Path(args.output_csv).as_posix()))


if __name__ == "__main__":
    main()
