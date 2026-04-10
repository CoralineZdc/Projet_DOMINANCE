import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


def pick_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    names = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        key = candidate.lower()
        if key in names:
            return names[key]
    return None


def to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value: str) -> Optional[int]:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def read_log(path: Path) -> Dict[str, List[Optional[float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Log not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")

        epoch_col = pick_column(reader.fieldnames, ["epoch"])
        train_loss_col = pick_column(reader.fieldnames, ["train_loss", "loss"])
        public_loss_col = pick_column(reader.fieldnames, ["public_loss", "val_loss", "valid_loss"])
        mean_col = pick_column(reader.fieldnames, ["pub_rmse_mean", "rmse_mean", "mean"])
        val_col = pick_column(reader.fieldnames, ["pub_rmse_val", "rmse_val", "valence", "v"])
        aro_col = pick_column(reader.fieldnames, ["pub_rmse_aro", "rmse_aro", "arousal", "a"])
        dom_col = pick_column(reader.fieldnames, ["pub_rmse_dom", "rmse_dom", "dominance", "d"])

        if epoch_col is None:
            raise ValueError(f"Missing epoch column in {path}")

        epochs: List[Optional[int]] = []
        train_loss: List[Optional[float]] = []
        public_loss: List[Optional[float]] = []
        mean: List[Optional[float]] = []
        val: List[Optional[float]] = []
        aro: List[Optional[float]] = []
        dom: List[Optional[float]] = []

        for row in reader:
            epochs.append(to_int(row.get(epoch_col, "")))
            train_loss.append(to_float(row.get(train_loss_col, "")) if train_loss_col else None)
            public_loss.append(to_float(row.get(public_loss_col, "")) if public_loss_col else None)
            mean.append(to_float(row.get(mean_col, "")) if mean_col else None)
            val.append(to_float(row.get(val_col, "")) if val_col else None)
            aro.append(to_float(row.get(aro_col, "")) if aro_col else None)
            dom.append(to_float(row.get(dom_col, "")) if dom_col else None)

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "public_loss": public_loss,
        "mean": mean,
        "val": val,
        "aro": aro,
        "dom": dom,
    }


def values_and_epochs(epochs: List[Optional[int]], values: List[Optional[float]]):
    xs = []
    ys = []
    for epoch, value in zip(epochs, values):
        if epoch is None or value is None:
            continue
        xs.append(epoch)
        ys.append(value)
    return xs, ys


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
            raise ValueError(f"Missing run_dir column in {summary_path}")

        for row in reader:
            run_dir_raw = str(row.get(run_dir_col, "")).strip()
            if not run_dir_raw:
                continue
            run_dir = Path(run_dir_raw.replace("\\", "/"))
            log_path = run_dir / "train_log.csv"
            if log_path.exists():
                logs.append(log_path)

    return logs


def gather_run_entries_from_summary(summary_path: Path) -> List[Tuple[str, Path]]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    entries: List[Tuple[str, Path]] = []
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return entries

        run_dir_col = pick_column(reader.fieldnames, ["run_dir"])
        run_id_col = pick_column(reader.fieldnames, ["run_id"])
        dataset_col = pick_column(reader.fieldnames, ["dataset"])
        status_col = pick_column(reader.fieldnames, ["status"])

        if run_dir_col is None:
            raise ValueError(f"Missing run_dir column in {summary_path}")

        for row in reader:
            if status_col and str(row.get(status_col, "")).strip().lower() not in {"", "ok"}:
                continue

            run_dir_raw = str(row.get(run_dir_col, "")).strip()
            if not run_dir_raw:
                continue

            run_dir = Path(run_dir_raw.replace("\\", "/"))
            log_path = run_dir / "train_log.csv"
            if not log_path.exists():
                continue

            run_id = str(row.get(run_id_col, "")).strip() if run_id_col else ""
            dataset = str(row.get(dataset_col, "")).strip() if dataset_col else ""
            label = run_dir.name
            if run_id and dataset:
                label = f"{run_id}:{dataset}"
            elif dataset:
                label = dataset
            elif run_id:
                label = f"run_{run_id}"

            entries.append((label, log_path))

    return entries


def plot_compare_mean(entries: List[Tuple[str, Path]], out_path: Path, title: str, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))

    plotted = 0
    for label, log_path in entries:
        series = read_log(log_path)
        xs, ys = values_and_epochs(series["epochs"], series["mean"])
        if not xs:
            continue
        ax.plot(xs, ys, linewidth=2.0, label=label)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError("No valid mean curves found to compare.")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean RMSE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved comparison plot to {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_single_log(log_path: Path, out_path: Path, title: str, show: bool) -> None:
    series = read_log(log_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    xs, ys = values_and_epochs(series["epochs"], series["train_loss"])
    if xs:
        axes[0].plot(xs, ys, label="train_loss", linewidth=2.0)

    xs, ys = values_and_epochs(series["epochs"], series["public_loss"])
    if xs:
        axes[0].plot(xs, ys, label="public_loss", linewidth=2.0)

    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    xs, ys = values_and_epochs(series["epochs"], series["mean"])
    if xs:
        axes[1].plot(xs, ys, label="mean", linewidth=2.2)

    xs, ys = values_and_epochs(series["epochs"], series["val"])
    if xs:
        axes[1].plot(xs, ys, label="V", linewidth=1.8)

    xs, ys = values_and_epochs(series["epochs"], series["aro"])
    if xs:
        axes[1].plot(xs, ys, label="A", linewidth=1.8)

    xs, ys = values_and_epochs(series["epochs"], series["dom"])
    if xs:
        axes[1].plot(xs, ys, label="D", linewidth=1.8)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved curve plot to {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from a log CSV file.")
    parser.add_argument(
        "log",
        nargs="?",
        default="",
        help="Path to log.csv or train_log.csv",
    )
    parser.add_argument(
        "--from-summary",
        type=str,
        default="",
        help="Optional gridsearch summary.csv to plot all run train_log.csv files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PNG path. Defaults to the log path with .png extension.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure window after saving.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional plot title.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="training_curve",
        help="Output filename prefix used with --from-summary (saved in each run folder).",
    )
    parser.add_argument(
        "--compare-mean",
        action="store_true",
        help="With --from-summary, create one overlay plot of mean RMSE for all runs.",
    )
    parser.add_argument(
        "--compare-out",
        type=str,
        default="",
        help="Output PNG path for --compare-mean. Default: <summary_dir>/compare_mean_rmse.png",
    )
    args = parser.parse_args()

    if args.from_summary:
        summary_path = Path(args.from_summary)
        logs = gather_logs_from_summary(summary_path)
        if not logs:
            raise SystemExit("No train_log.csv files found from summary.")

        if args.compare_mean:
            entries = gather_run_entries_from_summary(summary_path)
            if not entries:
                raise SystemExit("No valid run entries found to compare.")
            default_out = summary_path.parent / "compare_mean_rmse.png"
            out_path = Path(args.compare_out) if args.compare_out else default_out
            title = args.title.strip() or f"Mean RMSE Comparison ({summary_path.parent.name})"
            plot_compare_mean(entries, out_path, title, args.show)

        for log_path in logs:
            run_name = log_path.parent.name
            out_path = log_path.parent / f"{args.prefix}.png"
            title = args.title.strip() or run_name
            plot_single_log(log_path, out_path, title, show=False)
        print(f"Done. Plotted {len(logs)} runs.")
        return

    if not args.log:
        raise SystemExit("Provide a log path, or use --from-summary.")

    log_path = Path(args.log)
    out_path = Path(args.out) if args.out else log_path.with_suffix(".png")
    title = args.title.strip() or log_path.parent.name
    plot_single_log(log_path, out_path, title, args.show)


if __name__ == "__main__":
    main()
