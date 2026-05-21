from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import load_pima_dataset
from scripts.run_logistic import _run_one_dataset, settings_for_profile
from result_io import add_derived_mcmc_metrics, safe_to_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results" / "pima.csv"),
    )
    parser.add_argument("--tuning-profile", choices=("paper", "retuned"), default="paper")
    args = parser.parse_args()

    seed = args.seed
    output = Path(args.output)
    settings = settings_for_profile(args.tuning_profile)
    y, x = load_pima_dataset()
    print(
        json.dumps(
            {
                "event": "start",
                "dataset": "pima",
                "seed": seed,
                "settings": settings,
                "tuning_profile": args.tuning_profile,
                "output": str(output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    rows = _run_one_dataset("pima", y, x, settings, seed)
    frame = pd.DataFrame(rows)
    frame["tuning_profile"] = args.tuning_profile
    frame = add_derived_mcmc_metrics(frame, int(settings["n_samples"]))
    print(frame.to_string(index=False), flush=True)
    saved_to = safe_to_csv(frame, output)
    print(json.dumps({"event": "done", "output": str(saved_to)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
