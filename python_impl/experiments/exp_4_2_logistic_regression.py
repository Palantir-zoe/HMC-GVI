from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from python_impl.data import load_german_credit_dataset, load_pima_dataset
from python_impl.mcmc import adaptive_mh, hmc, mala, random_walk_mh
from python_impl.metrics import efficiency, lag1_autocorrelation
from python_impl.utils import now, set_seed
from python_impl.vi import cgvi
from python_impl.targets import logistic_grad_logposterior, logistic_logposterior


SETTINGS = {
    "quick": {"n_samples": 4000, "burn_in": 1000, "pre_burn": 1500},
    "paper": {"n_samples": 1000000, "burn_in": 10000, "pre_burn": 10000},
}


def _run_one_dataset(name: str, y: np.ndarray, x: np.ndarray, cfg: dict, seed: int) -> list[dict]:
    dim = x.shape[1]
    rng = set_seed(seed)

    logp = lambda beta: logistic_logposterior(beta, y, x)
    grad = lambda beta: logistic_grad_logposterior(beta, y, x)

    rows = []

    start = now()
    rmh_burn = random_walk_mh(logp, np.eye(dim) * 1e-3, cfg["pre_burn"], 0, rng)
    rmh_cov = 2.38**2 * np.cov(rmh_burn.samples, rowvar=False) / dim
    rmh = random_walk_mh(logp, rmh_cov, cfg["n_samples"], cfg["burn_in"], rng)
    runtime = now() - start
    rows.append(
        {
            "dataset": name,
            "method": "RMH",
            "time_sec": runtime,
            "efficiency": efficiency(rmh.samples),
            "p_jump": rmh.acceptance_rate,
            "rho1": lag1_autocorrelation(rmh.samples),
        }
    )

    start = now()
    am = adaptive_mh(logp, dim, cfg["n_samples"], cfg["burn_in"], rng)
    runtime = now() - start
    rows.append(
        {
            "dataset": name,
            "method": "AM",
            "time_sec": runtime,
            "efficiency": efficiency(am.samples),
            "p_jump": am.acceptance_rate,
            "rho1": lag1_autocorrelation(am.samples),
        }
    )

    start = now()
    mala_burn = mala(logp, grad, np.eye(dim), 0.20 if dim < 10 else 0.12, cfg["pre_burn"], 0, rng)
    mala_cov = np.cov(mala_burn.samples, rowvar=False)
    mala_step = 1.20 if name == "pima" else 1.63
    mala_run = mala(logp, grad, mala_cov, mala_step, cfg["n_samples"], cfg["burn_in"], rng)
    runtime = now() - start
    rows.append(
        {
            "dataset": name,
            "method": "MALA",
            "time_sec": runtime,
            "efficiency": efficiency(mala_run.samples),
            "p_jump": mala_run.acceptance_rate,
            "rho1": lag1_autocorrelation(mala_run.samples),
        }
    )

    start = now()
    hmc_burn = hmc(logp, grad, np.eye(dim), 0.10 if name == "pima" else 0.06, 20 if name == "pima" else 32, cfg["pre_burn"], 0, rng)
    hmc_cov = np.cov(hmc_burn.samples, rowvar=False)
    hmc_run = hmc(
        logp,
        grad,
        hmc_cov,
        0.15 if name == "pima" else 0.17,
        10,
        cfg["n_samples"],
        cfg["burn_in"],
        rng,
    )
    runtime = now() - start
    rows.append(
        {
            "dataset": name,
            "method": "HMC",
            "time_sec": runtime,
            "efficiency": efficiency(hmc_run.samples),
            "p_jump": hmc_run.acceptance_rate,
            "rho1": lag1_autocorrelation(hmc_run.samples),
        }
    )

    start = now()
    approx = cgvi(logp, grad, dim=dim, rng=rng)
    hmc_gvi = hmc(
        logp,
        grad,
        approx.covariance,
        0.16 if name == "pima" else 0.168,
        10,
        cfg["n_samples"],
        cfg["burn_in"],
        rng,
        init=approx.mean,
    )
    runtime = now() - start
    rows.append(
        {
            "dataset": name,
            "method": "HMC-GVI",
            "time_sec": runtime,
            "efficiency": efficiency(hmc_gvi.samples),
            "p_jump": hmc_gvi.acceptance_rate,
            "rho1": lag1_autocorrelation(hmc_gvi.samples),
        }
    )

    return rows


def run(mode: str, seed: int) -> pd.DataFrame:
    cfg = SETTINGS[mode]
    pima_y, pima_x = load_pima_dataset()
    german_y, german_x = load_german_credit_dataset()
    rows = []
    rows.extend(_run_one_dataset("pima", pima_y, pima_x, cfg, seed))
    rows.extend(_run_one_dataset("german", german_y, german_x, cfg, seed + 100))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "paper"], default="quick")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=str, default="python_impl/results_exp_4_2_logistic.csv")
    args = parser.parse_args()

    df = run(mode=args.mode, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(json.dumps({"saved_to": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
