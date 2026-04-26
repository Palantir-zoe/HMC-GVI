from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from python_impl.data import load_polypharm_dataset
from python_impl.mcmc import hmc, mala
from python_impl.metrics import efficiency, lag1_autocorrelation
from python_impl.targets import polypharm_grad_logposterior, polypharm_logposterior
from python_impl.utils import now, set_seed
from python_impl.vi import sparse_precision_gvi_polypharm


SETTINGS = {
    "quick": {
        "n_samples": 300,
        "burn_in": 80,
        "mala_pre_burn": 600,
        "hmc_pre_burn": 200,
        "gvi_max_iter": 400,
        "gvi_lb_window": 150,
        "gvi_max_patience": 8,
        "mala_gvi_epsilon": 0.10,
        "hmc_gvi_epsilon": 0.02,
    },
    "paper": {
        "n_samples": 1000000,
        "burn_in": 10000,
        "mala_pre_burn": 85000,
        "hmc_pre_burn": 100000,
        "mala_gvi_epsilon": 1.354,
        "hmc_gvi_epsilon": 0.185,
    },
}


def run(mode: str, seed: int) -> pd.DataFrame:
    cfg = SETTINGS[mode]
    rng = set_seed(seed)
    data = load_polypharm_dataset()
    dim = data["n_subjects"] + 8 + 1

    logp = lambda theta: polypharm_logposterior(theta, data)
    grad = lambda theta: polypharm_grad_logposterior(theta, data)

    rows = []

    start = now()
    mala_burn = mala(logp, grad, np.eye(dim), 0.15, cfg["mala_pre_burn"], 0, rng)
    burn_cov_mala = np.cov(mala_burn.samples, rowvar=False)
    mala_run = mala(logp, grad, burn_cov_mala, 0.522, cfg["n_samples"], cfg["burn_in"], rng)
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
    approx = sparse_precision_gvi_polypharm(
        logp,
        grad,
        dim=dim,
        random_effect_dim=data["n_subjects"],
        rng=rng,
        max_iter=cfg.get("gvi_max_iter", 20000),
        lb_window=cfg.get("gvi_lb_window", 10000),
        max_patience=cfg.get("gvi_max_patience", 50),
    )
    mala_gvi = mala(
        logp,
        grad,
        approx.covariance,
        cfg["mala_gvi_epsilon"],
        cfg["n_samples"],
        cfg["burn_in"],
        rng,
        init=approx.mean,
    )
    rows.append(
        {
            "method": "MALA-GVI",
            "time_sec": now() - start,
            "efficiency": efficiency(mala_gvi.samples),
            "p_jump": mala_gvi.acceptance_rate,
            "rho1": lag1_autocorrelation(mala_gvi.samples),
        }
    )

    start = now()
    hmc_burn = hmc(logp, grad, np.eye(dim), 0.03, 40, cfg["hmc_pre_burn"], 0, rng)
    burn_cov_hmc = np.cov(hmc_burn.samples, rowvar=False)
    hmc_run = hmc(logp, grad, burn_cov_hmc, 0.05, 20, cfg["n_samples"], cfg["burn_in"], rng)
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
    hmc_gvi = hmc(
        logp,
        grad,
        approx.covariance,
        cfg["hmc_gvi_epsilon"],
        10,
        cfg["n_samples"],
        cfg["burn_in"],
        rng,
        init=approx.mean,
    )
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
    parser.add_argument("--output", type=str, default="python_impl/results_exp_4_2_glmm.csv")
    args = parser.parse_args()

    df = run(mode=args.mode, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(json.dumps({"saved_to": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
