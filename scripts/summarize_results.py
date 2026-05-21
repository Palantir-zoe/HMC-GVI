from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RESULT_FILES = {
    "pima": "pima.csv",
    "german": "german.csv",
    "gaussian_100d": "gaussian_100d.csv",
    "glmm_polypharmacy": "glmm_full.csv",
}


SUMMARY_COLUMNS = [
    "dataset",
    "method",
    "tuning_profile",
    "time_sec",
    "preprocess_time_sec",
    "sampling_time_sec",
    "covariance_samples",
    "covariance_source",
    "burn_covariance_time_sec",
    "gvi_covariance_time_sec",
    "n_samples",
    "ess_mean",
    "efficiency",
    "p_jump",
    "rho1",
    "epsilon",
    "L",
]


def _load_one(input_dir: Path, dataset: str, filename: str) -> pd.DataFrame | None:
    path = input_dir / filename
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "dataset" not in frame.columns:
        frame.insert(0, "dataset", dataset)
    else:
        frame["dataset"] = frame["dataset"].fillna(dataset)
    for column in SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[SUMMARY_COLUMNS]


def _comparison_rows(summary: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    comparisons = [
        ("HMC-GVI", "HMC"),
        ("MALA-GVI", "MALA"),
    ]
    for dataset, group in summary.groupby("dataset", sort=False):
        by_method = {str(row["method"]): row for _, row in group.iterrows()}
        for method, baseline in comparisons:
            if method not in by_method or baseline not in by_method:
                continue
            current = by_method[method]
            base = by_method[baseline]
            eff_delta = float(current["efficiency"] - base["efficiency"])
            rho_delta = float(current["rho1"] - base["rho1"])
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "baseline": baseline,
                    "efficiency_delta": eff_delta,
                    "efficiency_ratio": float(current["efficiency"] / base["efficiency"])
                    if float(base["efficiency"]) != 0.0
                    else np.nan,
                    "ess_mean_delta": float(current["ess_mean"] - base["ess_mean"]),
                    "rho1_delta": rho_delta,
                    "time_sec_delta": float(current["time_sec"] - base["time_sec"]),
                    "gvi_or_method_better_efficiency": eff_delta > 0.0,
                    "gvi_or_method_lower_rho1": rho_delta < 0.0,
                }
            )
    return rows


def summarize(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    for dataset, filename in RESULT_FILES.items():
        frame = _load_one(input_dir, dataset, filename)
        if frame is not None:
            parts.append(frame)
    if not parts:
        raise FileNotFoundError(f"No known result CSVs found in {input_dir}")

    summary = pd.concat(parts, ignore_index=True)
    comparison = pd.DataFrame(_comparison_rows(summary))

    summary_path = input_dir / "summary_methods.csv"
    comparison_path = input_dir / "summary_comparisons.csv"
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    return summary, comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    summary, comparison = summarize(input_dir)
    print(summary.to_string(index=False))
    if not comparison.empty:
        print(comparison.to_string(index=False))
    print(
        json.dumps(
            {
                "summary": str(input_dir / "summary_methods.csv"),
                "comparisons": str(input_dir / "summary_comparisons.csv"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
