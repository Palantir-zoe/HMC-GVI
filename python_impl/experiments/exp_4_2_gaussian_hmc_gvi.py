from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from mcmc import adaptive_mh, hmc, mala, random_walk_mh
from metrics import efficiency, lag1_autocorrelation
from utils import draw_gaussian_covariance, now, set_seed
from vi import fit_fgvi_to_gaussian


SETTINGS = {
    "quick": {"n_samples": 300, "burn_in": 100, "pre_burn": 200},
    "paper": {"n_samples": 1000000, "burn_in": 10000, "pre_burn": 10000},
}


def run(mode: str, seed: int) -> pd.DataFrame:
    cfg = SETTINGS[mode]
    rng = set_seed(seed)
    dim = 100
    mean = np.zeros(dim, dtype=float)
    covariance = draw_gaussian_covariance(dim, rng)
    precision = np.linalg.inv(covariance)

    logp = lambda theta: float(-0.5 * (theta - mean) @ precision @ (theta - mean))
    grad = lambda theta: -(precision @ (theta - mean))

    rows = []

    start = now()
    rmh_burn = random_walk_mh(logp, np.eye(dim) * 1e-3, cfg["pre_burn"], 0, rng)
    rmh_cov = 2.38**2 * np.cov(rmh_burn.samples, rowvar=False) / dim
    rmh = random_walk_mh(logp, rmh_cov, cfg["n_samples"], cfg["burn_in"], rng)
    rows.append(
        {
            "method": "RMH",
            "time_sec": now() - start,
            "efficiency": efficiency(rmh.samples),
            "p_jump": rmh.acceptance_rate,
            "rho1": lag1_autocorrelation(rmh.samples),
        }
    )

    start = now()
    am = adaptive_mh(logp, dim, cfg["n_samples"], cfg["burn_in"], rng)
    rows.append(
        {
            "method": "AM",
            "time_sec": now() - start,
            "efficiency": efficiency(am.samples),
            "p_jump": am.acceptance_rate,
            "rho1": lag1_autocorrelation(am.samples),
        }
    )

    start = now()
    mala_burn = mala(logp, grad, np.eye(dim), 1.53, cfg["pre_burn"], 0, rng)
    mala_cov = np.cov(mala_burn.samples, rowvar=False)
    mala_run = mala(logp, grad, mala_cov, 1.515, cfg["n_samples"], cfg["burn_in"], rng)
    rows.append(
        {
            "method": "MALA",
            "time_sec": now() - start,
            "efficiency": efficiency(mala_run.samples),
            "p_jump": mala_run.acceptance_rate,
            "rho1": lag1_autocorrelation(mala_run.samples),
        }
    )

    start = now()
    hmc_burn = hmc(logp, grad, np.eye(dim), 0.16, 10, cfg["pre_burn"], 0, rng)
    hmc_cov = np.cov(hmc_burn.samples, rowvar=False)
    hmc_run = hmc(logp, grad, hmc_cov, 0.16, 10, cfg["n_samples"], cfg["burn_in"], rng)
    rows.append(
        {
            "method": "HMC",
            "time_sec": now() - start,
            "efficiency": efficiency(hmc_run.samples),
            "p_jump": hmc_run.acceptance_rate,
            "rho1": lag1_autocorrelation(hmc_run.samples),
        }
    )

    start = now()
    approx = fit_fgvi_to_gaussian(
        mean,
        covariance,
        rank=5,
        seed=seed,
        max_iter=150 if mode == "quick" else 1000,
    )
    hmc_gvi = hmc(logp, grad, approx.covariance, 0.16, 10, cfg["n_samples"], cfg["burn_in"], rng, init=approx.mean)
    rows.append(
        {
            "method": "HMC-GVI",
            "time_sec": now() - start,
            "efficiency": efficiency(hmc_gvi.samples),
            "p_jump": hmc_gvi.acceptance_rate,
            "rho1": lag1_autocorrelation(hmc_gvi.samples),
        }
    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "paper"], default="quick")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=str, default="python_impl/results_exp_4_2_gaussian.csv")
    args = parser.parse_args()

    df = run(mode=args.mode, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(json.dumps({"saved_to": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
