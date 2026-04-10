import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence


MEAN_CANDIDATES = ["pub_rmse_mean", "rmse_mean", "mean"]
VAL_CANDIDATES = ["pub_rmse_val", "rmse_val", "valence", "v"]
ARO_CANDIDATES = ["pub_rmse_aro", "rmse_aro", "arousal", "a"]
DOM_CANDIDATES = ["pub_rmse_dom", "rmse_dom", "dominance", "d"]
EPOCH_CANDIDATES = ["epoch"]


def pick_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    names = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in names:
            return names[candidate.lower()]
    return None


def to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value: str) -> Optional[int]:
    f = to_float(value)
    if f is None:
        return None
    return int(f)


def format_float(x: Optional[float], ndigits: int = 6) -> str:
    return "-" if x is None else f"{x:.{ndigits}f}"


def format_int(x: Optional[int]) -> str:
    return "-" if x is None else str(x)


def read_log(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Log not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")

        mean_col = pick_column(reader.fieldnames, MEAN_CANDIDATES)
        val_col = pick_column(reader.fieldnames, VAL_CANDIDATES)
        aro_col = pick_column(reader.fieldnames, ARO_CANDIDATES)
        dom_col = pick_column(reader.fieldnames, DOM_CANDIDATES)
        epoch_col = pick_column(reader.fieldnames, EPOCH_CANDIDATES)

        if mean_col is None or val_col is None or aro_col is None or dom_col is None:
            raise ValueError(
                f"Missing required columns in {path}. "
                f"Need mean/V/A/D columns (got: {reader.fieldnames})"
            )

        rows: List[Dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "epoch": to_int(row.get(epoch_col, "")) if epoch_col else None,
                    "mean": to_float(row.get(mean_col, "")),
                    "v": to_float(row.get(val_col, "")),
                    "a": to_float(row.get(aro_col, "")),
                    "d": to_float(row.get(dom_col, "")),
                }
            )

    valid_rows = [r for r in rows if r["mean"] is not None]
    if not valid_rows:
        raise ValueError(f"No valid mean values found in {path}")

    best_row = min(valid_rows, key=lambda r: r["mean"])
    last_row = valid_rows[-1]

    return {
        "path": path,
        "rows": rows,
        "best_row": best_row,
        "last_row": last_row,
    }


def gather_logs_from_summary(summary_path: Path) -> List[Path]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    logs: List[Path] = []
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return logs
        run_dir_col = pick_column(reader.fieldnames, ["run_dir"])
        if run_dir_col is None:
            return logs

        for row in reader:
            run_dir_raw = str(row.get(run_dir_col, "")).strip()
            if not run_dir_raw:
                continue
            run_dir = Path(run_dir_raw.replace("\\", "/"))
            log_path = run_dir / "train_log.csv"
            if log_path.exists():
                logs.append(log_path)

    return logs


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def print_run_summary(log_infos: Sequence[Dict[str, object]]) -> None:
    rows: List[List[str]] = []
    for info in log_infos:
        path = info["path"]
        best = info["best_row"]
        last = info["last_row"]
        rows.append(
            [
                Path(path).parent.name,
                format_int(best["epoch"]),
                format_float(best["mean"]),
                format_float(best["v"]),
                format_float(best["a"]),
                format_float(best["d"]),
                format_int(last["epoch"]),
                format_float(last["mean"]),
                format_float(last["v"]),
                format_float(last["a"]),
                format_float(last["d"]),
            ]
        )

    headers = [
        "run",
        "best_ep",
        "best_mean",
        "best_V",
        "best_A",
        "best_D",
        "last_ep",
        "last_mean",
        "last_V",
        "last_A",
        "last_D",
    ]
    print_table(headers, rows)


def print_per_epoch(log_info: Dict[str, object], tail: int) -> None:
    rows = [r for r in log_info["rows"] if r["mean"] is not None]
    if tail > 0:
        rows = rows[-tail:]

    table_rows: List[List[str]] = []
    for r in rows:
        table_rows.append(
            [
                format_int(r["epoch"]),
                format_float(r["mean"]),
                format_float(r["v"]),
                format_float(r["a"]),
                format_float(r["d"]),
            ]
        )

    print(f"\n[{log_info['path']}]")
    print_table(["epoch", "mean", "V", "A", "D"], table_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display mean/V/A/D losses neatly from train log CSV files."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        help="Paths to log CSV files (for example FER2013_ResNet/log.csv or run/train_log.csv).",
    )
    parser.add_argument(
        "--from-summary",
        type=str,
        default="",
        help="Optional gridsearch summary.csv to auto-load each run's train_log.csv.",
    )
    parser.add_argument(
        "--per-epoch",
        action="store_true",
        help="Also print per-epoch mean/V/A/D table for each log.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=15,
        help="Number of last epochs to show with --per-epoch (0 means all).",
    )
    args = parser.parse_args()

    log_paths: List[Path] = [Path(p) for p in args.logs]

    if args.from_summary:
        log_paths.extend(gather_logs_from_summary(Path(args.from_summary)))

    # Preserve order while removing duplicates.
    unique_paths: List[Path] = []
    seen = set()
    for p in log_paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            unique_paths.append(p)
            seen.add(key)

    if not unique_paths:
        raise SystemExit("No logs provided. Pass log paths or --from-summary.")

    infos: List[Dict[str, object]] = []
    for p in unique_paths:
        infos.append(read_log(p))

    print_run_summary(infos)

    if args.per_epoch:
        for info in infos:
            print_per_epoch(info, args.tail)


if __name__ == "__main__":
    main()
