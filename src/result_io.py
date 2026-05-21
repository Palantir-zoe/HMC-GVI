from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def timestamped_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def safe_to_csv(frame: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_csv(output, index=False)
        return output
    except PermissionError:
        fallback = timestamped_path(output)
        frame.to_csv(fallback, index=False)
        return fallback


def prepare_overwrite_output(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        return output
    try:
        output.unlink()
        return output
    except PermissionError:
        return timestamped_path(output)


def add_derived_mcmc_metrics(frame: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    frame = frame.copy()
    if "n_samples" not in frame.columns:
        frame["n_samples"] = int(n_samples)
    if "efficiency" in frame.columns and "ess_mean" not in frame.columns:
        frame["ess_mean"] = frame["efficiency"] * frame["n_samples"]
    return frame


def add_derived_mcmc_metrics_to_row(row: dict[str, object], n_samples: int) -> dict[str, object]:
    row = dict(row)
    row.setdefault("n_samples", int(n_samples))
    if "efficiency" in row and "ess_mean" not in row:
        row["ess_mean"] = float(row["efficiency"]) * int(row["n_samples"])
    return row


def append_row_to_csv(row: dict[str, object], output: Path, n_samples: int) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    row = add_derived_mcmc_metrics_to_row(row, n_samples)
    frame = pd.DataFrame([row])
    frame.to_csv(output, mode="a", header=not output.exists(), index=False)
    return row
