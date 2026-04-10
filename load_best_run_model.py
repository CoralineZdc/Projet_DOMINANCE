import argparse
import csv
from pathlib import Path

import torch

from compute_rmse import build_model_from_state_dict


def load_state_dict(model_dir: Path):
    candidates = [
        model_dir / "best_model_state.pth",
        model_dir / "best_model.pth",
        model_dir / "checkpoint.pth",
        model_dir / "last_model.pth",
    ]

    for path in candidates:
        if not path.exists():
            continue

        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            return state["model"], path
        return state, path

    raise FileNotFoundError("No model weights found in {}".format(model_dir))


def find_best_run(summary_path: Path):
    if not summary_path.exists():
        raise FileNotFoundError("Summary CSV not found: {}".format(summary_path))

    best_row = None
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") not in {"ok", "done"}:
                continue
            raw_score = row.get("best_pub_rmse_mean")
            if raw_score in {None, "", "NA"}:
                continue
            try:
                score = float(raw_score)
            except ValueError:
                continue

            if best_row is None or score < float(best_row["best_pub_rmse_mean"]):
                best_row = row

    if best_row is None:
        raise RuntimeError("No completed runs with a valid best_pub_rmse_mean were found in {}".format(summary_path))

    return best_row


def main():
    parser = argparse.ArgumentParser(description="Load the best trained model from a grid-search summary")
    parser.add_argument(
        "--summary",
        default="gridsearch_runs/grid-20260409-150411/summary.csv",
        help="Path to a sweep summary CSV",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to resolve relative run_dir paths",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to place the model on after loading",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    repo_root = Path(args.repo_root)

    best_row = find_best_run(summary_path)
    run_dir = repo_root / Path(best_row["run_dir"])

    state_dict, weights_path = load_state_dict(run_dir)
    model = build_model_from_state_dict(state_dict)
    model.load_state_dict(state_dict)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        model = model.cuda()

    print("Best run loaded")
    print("Dataset: {}".format(best_row.get("dataset")))
    print("Summary row best_pub_rmse_mean: {}".format(best_row.get("best_pub_rmse_mean")))
    print("Run dir: {}".format(run_dir))
    print("Weights file: {}".format(weights_path))
    print("Model class: {}".format(model.__class__.__name__))
    print("Trainable parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    return model


if __name__ == "__main__":
    main()