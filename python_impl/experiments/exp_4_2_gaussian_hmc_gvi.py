from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from python_impl.mcmc import (
    adaptive_mh,
    hmc,
    mala,
    random_walk_mh,
)
from python_impl.metrics import efficiency, lag1_autocorrelation
from python_impl.tuning import tune_hmc_hyperparameters, tune_mala_epsilon
from python_impl.utils import draw_gaussian_covariance, now, regularized_sample_covariance, set_seed
from python_impl.vi import fit_fgvi_to_gaussian


SETTINGS = {
    "n_samples": 1000000,
    "burn_in": 10000,
    "pre_burn": 10000,
    "mala_tune_rounds": 8,
    "mala_tune_steps": 200,
    "hmc_tune_rounds": 6,
    "hmc_tune_steps": 100,
    "hmc_eval_steps": 200,
    "hmc_leapfrog_candidates": (8, 10, 12, 16, 20, 24, 32),
    "burn_cov_shrinkage": 0.0,
}


def run(seed: int) -> pd.DataFrame:
    cfg = SETTINGS
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
            "covariance_samples": int(rmh_burn.samples.shape[0]),
            "covariance_source": "rmh_pre_burn",
            "efficiency": efficiency(rmh.samples),
            "p_jump": rmh.acceptance_rate,
            "rho1": lag1_autocorrelation(rmh.samples),
            "epsilon": np.nan,
            "L": np.nan,
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
            "epsilon": np.nan,
            "L": np.nan,
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
    rows.append(
        {
            "method": "MALA",
            "time_sec": now() - start,
            "covariance_samples": int(mala_burn.samples.shape[0]),
            "covariance_source": "mala_pre_burn",
            "efficiency": efficiency(mala_run.samples),
            "p_jump": mala_run.acceptance_rate,
            "rho1": lag1_autocorrelation(mala_run.samples),
            "epsilon": mala_main_tune.epsilon,
            "L": np.nan,
        }
    )

    start = now()
    hmc_burn_tune = tune_hmc_hyperparameters(
        logp,
        grad,
        np.eye(dim),
        rng,
        initial_epsilon=0.16,
        leapfrog_candidates=cfg["hmc_leapfrog_candidates"],
        target_acceptance=0.95,
        min_acceptance=0.80,
        tuning_rounds=cfg["hmc_tune_rounds"],
        tuning_steps=cfg["hmc_tune_steps"],
        evaluation_steps=cfg["hmc_eval_steps"],
    )
    hmc_tune1_time = now() - start
    hmc_pre_start = now()
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
    hmc_pre_time = now() - hmc_pre_start
    hmc_cov_start = now()
    hmc_cov = regularized_sample_covariance(
        hmc_burn.samples,
        np.eye(dim),
        shrinkage=cfg["burn_cov_shrinkage"],
    )
    hmc_cov_time = now() - hmc_cov_start
    hmc_tune2_start = now()
    hmc_main_tune = tune_hmc_hyperparameters(
        logp,
        grad,
        hmc_cov,
        rng,
        init=hmc_burn.samples[-1],
        initial_epsilon=hmc_burn_tune.epsilon,
        leapfrog_candidates=cfg["hmc_leapfrog_candidates"],
        target_acceptance=0.95,
        min_acceptance=0.80,
        tuning_rounds=cfg["hmc_tune_rounds"],
        tuning_steps=cfg["hmc_tune_steps"],
        evaluation_steps=cfg["hmc_eval_steps"],
    )
    hmc_tune2_time = now() - hmc_tune2_start
    hmc_main_start = now()
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
    hmc_main_time = now() - hmc_main_start
    rows.append(
        {
            "method": "HMC",
            "time_sec": now() - start,
            "preprocess_time_sec": hmc_tune1_time + hmc_pre_time + hmc_cov_time + hmc_tune2_time,
            "sampling_time_sec": hmc_main_time,
            "covariance_samples": int(hmc_burn.samples.shape[0]),
            "covariance_source": "hmc_pre_burn",
            "efficiency": efficiency(hmc_run.samples),
            "p_jump": hmc_run.acceptance_rate,
            "rho1": lag1_autocorrelation(hmc_run.samples),
            "epsilon": hmc_main_tune.epsilon,
            "L": int(hmc_main_tune.n_leapfrog),
        }
    )

    start = now()
    gvi_start = now()
    approx = fit_fgvi_to_gaussian(
        mean,
        covariance,
        rank=5,
        seed=seed,
        max_iter=1000,
    )
    gvi_time = now() - gvi_start
    hmc_gvi_tune_start = now()
    hmc_gvi_tune = tune_hmc_hyperparameters(
        logp,
        grad,
        approx.covariance,
        rng,
        init=approx.mean,
        initial_epsilon=hmc_main_tune.epsilon,
        leapfrog_candidates=cfg["hmc_leapfrog_candidates"],
        target_acceptance=0.95,
        min_acceptance=0.80,
        tuning_rounds=cfg["hmc_tune_rounds"],
        tuning_steps=cfg["hmc_tune_steps"],
        evaluation_steps=cfg["hmc_eval_steps"],
    )
    hmc_gvi_tune_time = now() - hmc_gvi_tune_start
    hmc_gvi_main_start = now()
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
    hmc_gvi_main_time = now() - hmc_gvi_main_start
    rows.append(
        {
            "method": "HMC-GVI",
            "time_sec": gvi_time + now() - start,
            "preprocess_time_sec": gvi_time + hmc_gvi_tune_time,
            "sampling_time_sec": hmc_gvi_main_time,
            "covariance_samples": np.nan,
            "covariance_source": "fgvi",
            "efficiency": efficiency(hmc_gvi.samples),
            "p_jump": hmc_gvi.acceptance_rate,
            "rho1": lag1_autocorrelation(hmc_gvi.samples),
            "epsilon": hmc_gvi_tune.epsilon,
            "L": int(hmc_gvi_tune.n_leapfrog),
        }
    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results_exp_4_2_gaussian.csv"),
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
