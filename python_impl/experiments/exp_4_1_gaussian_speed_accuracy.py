from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python_impl.metrics import gaussian_moment_vector, rmse
from python_impl.mcmc import metropolis_within_gibbs_gaussian
from python_impl.utils import now, set_seed
from python_impl.vi import fit_fgvi_to_gaussian


PAPER_DIMS = [2, 10, 50, 100, 500]


def run(seed: int) -> pd.DataFrame:
    rng = set_seed(seed)
    rows = []

    for dim in PAPER_DIMS:
        mean = np.zeros(dim, dtype=float)
        covariance = np.eye(dim, dtype=float)
        true_vector = gaussian_moment_vector(mean, covariance)

        rank = 1
        start = now()
        fgvi = fit_fgvi_to_gaussian(mean, covariance, rank=rank, seed=seed + dim)
        fgvi_time = now() - start
        fgvi_vector = gaussian_moment_vector(fgvi.mean, fgvi.covariance)
        fgvi_rmse = rmse(true_vector, fgvi_vector)

        step_size = 0.75
        mcmc = metropolis_within_gibbs_gaussian(
            mean=mean,
            covariance=covariance,
            step_size=step_size,
            time_budget_seconds=fgvi_time,
            rng=rng,
            min_iterations=max(30, dim + 2),
        )
        mcmc_mean = mcmc.samples.mean(axis=0)
        mcmc_cov = np.cov(mcmc.samples, rowvar=False)
        mcmc_vector = gaussian_moment_vector(mcmc_mean, mcmc_cov)
        mcmc_rmse = rmse(true_vector, mcmc_vector)

        rows.append(
            {
                "dimension": dim,
                "fgvi_rmse": fgvi_rmse,
                "mcmc_rmse": mcmc_rmse,
                "fgvi_time_sec": fgvi_time,
                "mcmc_acceptance_rate": mcmc.acceptance_rate,
                "mcmc_iterations": int(mcmc.samples.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results_exp_4_1.csv"),
    )
    args = parser.parse_args()

    df = run(seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(df.to_string(index=False))
    print(json.dumps({"saved_to": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
