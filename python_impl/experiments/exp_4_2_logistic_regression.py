from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python_impl.data import load_german_credit_dataset, load_pima_dataset
from python_impl.mcmc import (
    adaptive_mh,
    hmc,
    mala,
    random_walk_mh,
)
from python_impl.metrics import efficiency, lag1_autocorrelation
from python_impl.targets import logistic_grad_logposterior, logistic_logposterior
from python_impl.tuning import tune_hmc_hyperparameters, tune_mala_epsilon
from python_impl.utils import now, set_seed
from python_impl.vi import cgvi


SETTINGS = {
    "n_samples": 1000000,
    "burn_in": 10000,
    "pre_burn": 10000,
    "mala_tune_rounds": 8,
    "mala_tune_steps": 200,
    "hmc_tune_rounds": 6,
    "hmc_tune_steps": 100,
    "hmc_eval_steps": 200,
    "hmc_leapfrog_candidates": (8, 12, 16, 24, 32),
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
            "covariance_samples": int(rmh_burn.samples.shape[0]),
            "covariance_source": "rmh_pre_burn",
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
            "covariance_samples": np.nan,
            "covariance_source": "adaptive",
            "efficiency": efficiency(am.samples),
            "p_jump": am.acceptance_rate,
            "rho1": lag1_autocorrelation(am.samples),
        }
    )

    start = now()
    mala_burn_tune = tune_mala_epsilon(
        logp,
        grad,
        np.eye(dim),
        rng,
        initial_epsilon=1.65,
        tuning_rounds=cfg["mala_tune_rounds"],
        tuning_steps=cfg["mala_tune_steps"],
    )
    mala_burn = mala(
        logp,
        grad,
        np.eye(dim),
        mala_burn_tune.epsilon,
        cfg["pre_burn"],
        0,
        rng,
        init=mala_burn_tune.final_state,
    )
    mala_cov = np.cov(mala_burn.samples, rowvar=False)
    mala_main_tune = tune_mala_epsilon(
        logp,
        grad,
        mala_cov,
        rng,
        init=mala_burn.samples[-1],
        initial_epsilon=mala_burn_tune.epsilon,
        tuning_rounds=cfg["mala_tune_rounds"],
        tuning_steps=cfg["mala_tune_steps"],
    )
    mala_run = mala(
        logp,
        grad,
        mala_cov,
        mala_main_tune.epsilon,
        cfg["n_samples"],
        cfg["burn_in"],
        rng,
        init=mala_main_tune.final_state,
    )
    runtime = now() - start
    rows.append(
        {
            "dataset": name,
            "method": "MALA",
            "time_sec": runtime,
            "covariance_samples": int(mala_burn.samples.shape[0]),
            "covariance_source": "mala_pre_burn",
            "efficiency": efficiency(mala_run.samples),
            "p_jump": mala_run.acceptance_rate,
            "rho1": lag1_autocorrelation(mala_run.samples),
        }
    )

    start = now()
    hmc_burn_tune = tune_hmc_hyperparameters(
        logp,
        grad,
        np.eye(dim),
        rng,
        initial_epsilon=0.10 if name == "pima" else 0.06,
        leapfrog_candidates=cfg["hmc_leapfrog_candidates"],
        tuning_rounds=cfg["hmc_tune_rounds"],
        tuning_steps=cfg["hmc_tune_steps"],
        evaluation_steps=cfg["hmc_eval_steps"],
    )
    hmc_burn = hmc(
        logp,
        grad,
        np.eye(dim),
        hmc_burn_tune.epsilon,
        int(hmc_burn_tune.n_leapfrog),
        cfg["pre_burn"],
        0,
        rng,
        init=hmc_burn_tune.final_state,
    )
    hmc_cov = np.cov(hmc_burn.samples, rowvar=False)
    hmc_main_tune = tune_hmc_hyperparameters(
        logp,
        grad,
        hmc_cov,
        rng,
        init=hmc_burn.samples[-1],
        initial_epsilon=hmc_burn_tune.epsilon,
        leapfrog_candidates=cfg["hmc_leapfrog_candidates"],
        tuning_rounds=cfg["hmc_tune_rounds"],
        tuning_steps=cfg["hmc_tune_steps"],
        evaluation_steps=cfg["hmc_eval_steps"],
    )
    hmc_run = hmc(
        logp,
        grad,
        hmc_cov,
        hmc_main_tune.epsilon,
        int(hmc_main_tune.n_leapfrog),
        cfg["n_samples"],
        cfg["burn_in"],
        rng,
        init=hmc_main_tune.final_state,
    )
    runtime = now() - start
    rows.append(
        {
            "dataset": name,
            "method": "HMC",
            "time_sec": runtime,
            "covariance_samples": int(hmc_burn.samples.shape[0]),
            "covariance_source": "hmc_pre_burn",
            "efficiency": efficiency(hmc_run.samples),
            "p_jump": hmc_run.acceptance_rate,
            "rho1": lag1_autocorrelation(hmc_run.samples),
        }
    )

    start = now()
    gvi_start = now()
    approx = cgvi(logp, grad, dim=dim, rng=rng)
    gvi_time = now() - gvi_start
    hmc_gvi_tune = tune_hmc_hyperparameters(
        logp,
        grad,
        approx.covariance,
        rng,
        init=approx.mean,
        initial_epsilon=hmc_main_tune.epsilon,
        leapfrog_candidates=cfg["hmc_leapfrog_candidates"],
        tuning_rounds=cfg["hmc_tune_rounds"],
        tuning_steps=cfg["hmc_tune_steps"],
        evaluation_steps=cfg["hmc_eval_steps"],
    )
    hmc_gvi = hmc(
        logp,
        grad,
        approx.covariance,
        hmc_gvi_tune.epsilon,
        int(hmc_gvi_tune.n_leapfrog),
        cfg["n_samples"],
        cfg["burn_in"],
        rng,
        init=hmc_gvi_tune.final_state,
    )
    runtime = now() - start
    rows.append(
        {
            "dataset": name,
            "method": "HMC-GVI",
            "time_sec": gvi_time + runtime,
            "covariance_samples": np.nan,
            "covariance_source": "cgvi",
            "efficiency": efficiency(hmc_gvi.samples),
            "p_jump": hmc_gvi.acceptance_rate,
            "rho1": lag1_autocorrelation(hmc_gvi.samples),
        }
    )

    return rows


def run(seed: int) -> pd.DataFrame:
    cfg = SETTINGS
    pima_y, pima_x = load_pima_dataset()
    german_y, german_x = load_german_credit_dataset()
    rows = []
    rows.extend(_run_one_dataset("pima", pima_y, pima_x, cfg, seed))
    rows.extend(_run_one_dataset("german", german_y, german_x, cfg, seed + 100))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results_exp_4_2_logistic.csv"),
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
