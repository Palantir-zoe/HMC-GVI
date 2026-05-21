from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import load_polypharm_dataset
from mcmc import hmc, mala
from metrics import efficiency, lag1_autocorrelation
from targets import polypharm_grad_logposterior, polypharm_logposterior
from tuning import tune_hmc_hyperparameters, tune_mala_epsilon
from utils import now, regularized_sample_covariance, set_seed
from vi import sparse_precision_gvi_polypharm
from result_io import add_derived_mcmc_metrics, safe_to_csv


SETTINGS = {
    "n_samples": 1000000,
    "burn_in": 10000,
    "pre_burn": 10000,
    "burn_cov_pre_burn": 100000,
    "gvi_max_iter": 20000,
    "gvi_lb_window": 10000,
    "gvi_max_patience": 50,
    "mala_tune_rounds": 8,
    "mala_tune_steps": 150,
    "hmc_tune_rounds": 6,
    "hmc_tune_steps": 80,
    "hmc_eval_steps": 160,
    "hmc_burn_leapfrog_candidates": (20, 40),
    "hmc_main_leapfrog_candidates": (10, 20, 40),
    "hmc_gvi_leapfrog_candidates": (8, 10, 12, 16, 20),
    "hmc_burn_initial_epsilon": 0.03,
    "hmc_main_initial_epsilon": 0.05,
    "hmc_gvi_initial_epsilon": 0.185,
    "burn_cov_shrinkage": 0.0,
}


RETUNED_SETTINGS = {
    "hmc_tune_rounds": 7,
    "hmc_tune_steps": 100,
    "hmc_eval_steps": 240,
    "hmc_burn_leapfrog_candidates": (16, 20, 32, 40),
    "hmc_main_leapfrog_candidates": (8, 10, 12, 16, 20, 24, 32, 40),
    "hmc_gvi_leapfrog_candidates": (8, 10, 12, 16, 20, 24, 32, 40),
}


def settings_for_profile(profile: str) -> dict:
    cfg = dict(SETTINGS)
    if profile == "retuned":
        cfg.update(RETUNED_SETTINGS)
    return cfg


def _safe_covariance(samples: np.ndarray, fallback: np.ndarray, shrinkage: float | None) -> np.ndarray:
    return regularized_sample_covariance(samples, fallback, shrinkage=shrinkage)


def run(seed: int, cfg: dict | None = None) -> pd.DataFrame:
    cfg = SETTINGS if cfg is None else cfg
    rng = set_seed(seed)
    data = load_polypharm_dataset()
    dim = data["n_subjects"] + 8 + 1

    logp = lambda theta: polypharm_logposterior(theta, data)
    grad = lambda theta: polypharm_grad_logposterior(theta, data)

    rows = []

    start = now()
    mala_cov_start = now()
    burn_cov_pre_burn = int(cfg.get("burn_cov_pre_burn", cfg["pre_burn"]))
    mala_burn_tune = tune_mala_epsilon(
        logp,
        grad,
        np.eye(dim),
        rng,
        initial_epsilon=0.15,
        tuning_rounds=cfg["mala_tune_rounds"],
        tuning_steps=cfg["mala_tune_steps"],
        epsilon_bounds=(1e-3, 1.0),
    )
    mala_burn = mala(
        logp,
        grad,
        np.eye(dim),
        mala_burn_tune.epsilon,
        burn_cov_pre_burn,
        0,
        rng,
        init=mala_burn_tune.final_state,
    )
    burn_cov_mala = _safe_covariance(mala_burn.samples, np.eye(dim), cfg["burn_cov_shrinkage"])
    mala_cov_time = now() - mala_cov_start
    mala_covariance_samples = int(mala_burn.samples.shape[0])
    mala_burn_final = mala_burn.samples[-1].copy()
    mala_main_tune_start = now()
    mala_main_tune = tune_mala_epsilon(
        logp,
        grad,
        burn_cov_mala,
        rng,
        init=mala_burn_final,
        initial_epsilon=mala_burn_tune.epsilon,
        tuning_rounds=cfg["mala_tune_rounds"],
        tuning_steps=cfg["mala_tune_steps"],
        epsilon_bounds=(1e-3, 1.0),
    )
    mala_main_tune_time = now() - mala_main_tune_start
    del mala_burn
    gc.collect()
    mala_main_start = now()
    mala_run = mala(
        logp,
        grad,
        burn_cov_mala,
        mala_main_tune.epsilon,
        cfg["n_samples"],
        cfg["burn_in"],
        rng,
        init=mala_main_tune.final_state,
    )
    mala_main_time = now() - mala_main_start
    rows.append(
        {
            "method": "MALA",
            "time_sec": now() - start,
            "preprocess_time_sec": mala_cov_time + mala_main_tune_time,
            "sampling_time_sec": mala_main_time,
            "covariance_samples": mala_covariance_samples,
            "covariance_source": "mala_pre_burn",
            "burn_covariance_time_sec": mala_cov_time,
            "gvi_covariance_time_sec": np.nan,
            "efficiency": efficiency(mala_run.samples),
            "p_jump": mala_run.acceptance_rate,
            "rho1": lag1_autocorrelation(mala_run.samples),
            "epsilon": mala_main_tune.epsilon,
            "L": np.nan,
        }
    )

    start = now()
    gvi_fit_start = now()
    approx = sparse_precision_gvi_polypharm(
        logp,
        grad,
        dim=dim,
        random_effect_dim=data["n_subjects"],
        rng=rng,
        max_iter=cfg["gvi_max_iter"],
        lb_window=cfg["gvi_lb_window"],
        max_patience=cfg["gvi_max_patience"],
    )
    gvi_time = now() - gvi_fit_start
    gvi_source = "sparse_precision_gvi"
    mala_gvi_tune_start = now()
    mala_gvi_tune = tune_mala_epsilon(
        logp,
        grad,
        approx.covariance,
        rng,
        init=approx.mean,
        initial_epsilon=mala_main_tune.epsilon,
        tuning_rounds=cfg["mala_tune_rounds"],
        tuning_steps=cfg["mala_tune_steps"],
        epsilon_bounds=(1e-3, 1.0),
    )
    mala_gvi_tune_time = now() - mala_gvi_tune_start
    mala_gvi_main_start = now()
    mala_gvi = mala(
        logp,
        grad,
        approx.covariance,
        mala_gvi_tune.epsilon,
        cfg["n_samples"],
        cfg["burn_in"],
        rng,
        init=mala_gvi_tune.final_state,
    )
    mala_gvi_main_time = now() - mala_gvi_main_start
    rows.append(
        {
            "method": "MALA-GVI",
            "time_sec": now() - start,
            "preprocess_time_sec": gvi_time + mala_gvi_tune_time,
            "sampling_time_sec": mala_gvi_main_time,
            "covariance_samples": np.nan,
            "covariance_source": gvi_source,
            "burn_covariance_time_sec": np.nan,
            "gvi_covariance_time_sec": gvi_time,
            "efficiency": efficiency(mala_gvi.samples),
            "p_jump": mala_gvi.acceptance_rate,
            "rho1": lag1_autocorrelation(mala_gvi.samples),
            "epsilon": mala_gvi_tune.epsilon,
            "L": np.nan,
        }
    )

    start = now()
    burn_cov_pre_burn = int(cfg.get("burn_cov_pre_burn", cfg["pre_burn"]))
    hmc_burn_tune = tune_hmc_hyperparameters(
        logp,
        grad,
        np.eye(dim),
        rng,
        initial_epsilon=cfg["hmc_burn_initial_epsilon"],
        leapfrog_candidates=cfg["hmc_burn_leapfrog_candidates"],
        target_acceptance=0.95,
        min_acceptance=0.80,
        tuning_rounds=cfg["hmc_tune_rounds"],
        tuning_steps=cfg["hmc_tune_steps"],
        evaluation_steps=cfg["hmc_eval_steps"],
        epsilon_bounds=(1e-4, 0.06),
    )
    hmc_tune1_time = now() - start
    hmc_pre_start = now()
    hmc_burn = hmc(
        logp,
        grad,
        np.eye(dim),
        hmc_burn_tune.epsilon,
        int(hmc_burn_tune.n_leapfrog),
        burn_cov_pre_burn,
        0,
        rng,
        init=hmc_burn_tune.final_state,
    )
    hmc_pre_time = now() - hmc_pre_start
    hmc_cov_start = now()
    burn_cov_hmc = _safe_covariance(hmc_burn.samples, np.eye(dim), cfg["burn_cov_shrinkage"])
    hmc_cov_time = now() - hmc_cov_start
    hmc_covariance_samples = int(hmc_burn.samples.shape[0])
    hmc_burn_final = hmc_burn.samples[-1].copy()
    hmc_tune2_start = now()
    hmc_main_tune = tune_hmc_hyperparameters(
        logp,
        grad,
        burn_cov_hmc,
        rng,
        init=hmc_burn_final,
        initial_epsilon=cfg["hmc_main_initial_epsilon"],
        leapfrog_candidates=cfg["hmc_main_leapfrog_candidates"],
        target_acceptance=0.95,
        min_acceptance=0.80,
        tuning_rounds=cfg["hmc_tune_rounds"],
        tuning_steps=cfg["hmc_tune_steps"],
        evaluation_steps=cfg["hmc_eval_steps"],
        epsilon_bounds=(1e-4, 0.12),
    )
    hmc_tune2_time = now() - hmc_tune2_start
    del hmc_burn
    gc.collect()
    hmc_epsilon = hmc_main_tune.epsilon
    hmc_leapfrog = int(hmc_main_tune.n_leapfrog)
    hmc_init = hmc_main_tune.final_state
    hmc_main_start = now()
    hmc_run = hmc(
        logp,
        grad,
        burn_cov_hmc,
        hmc_epsilon,
        hmc_leapfrog,
        cfg["n_samples"],
        cfg["burn_in"],
        rng,
        init=hmc_init,
    )
    hmc_main_time = now() - hmc_main_start
    rows.append(
        {
            "method": "HMC",
            "time_sec": now() - start,
            "preprocess_time_sec": hmc_tune1_time + hmc_pre_time + hmc_cov_time + hmc_tune2_time,
            "sampling_time_sec": hmc_main_time,
            "covariance_samples": hmc_covariance_samples,
            "covariance_source": "hmc_pre_burn",
            "burn_covariance_time_sec": hmc_tune1_time + hmc_pre_time + hmc_cov_time,
            "gvi_covariance_time_sec": np.nan,
            "efficiency": efficiency(hmc_run.samples),
            "p_jump": hmc_run.acceptance_rate,
            "rho1": lag1_autocorrelation(hmc_run.samples),
            "epsilon": hmc_epsilon,
            "L": hmc_leapfrog,
        }
    )

    start = now()
    hmc_gvi_tune_start = now()
    hmc_gvi_tune = tune_hmc_hyperparameters(
        logp,
        grad,
        approx.covariance,
        rng,
        init=approx.mean,
        initial_epsilon=cfg["hmc_gvi_initial_epsilon"],
        leapfrog_candidates=cfg["hmc_gvi_leapfrog_candidates"],
        target_acceptance=0.95,
        min_acceptance=0.80,
        tuning_rounds=cfg["hmc_tune_rounds"],
        tuning_steps=cfg["hmc_tune_steps"],
        evaluation_steps=cfg["hmc_eval_steps"],
        epsilon_bounds=(1e-4, 0.25),
    )
    hmc_gvi_tune_time = now() - hmc_gvi_tune_start
    hmc_gvi_epsilon = hmc_gvi_tune.epsilon
    hmc_gvi_leapfrog = int(hmc_gvi_tune.n_leapfrog)
    hmc_gvi_init = hmc_gvi_tune.final_state
    hmc_gvi_main_start = now()
    hmc_gvi = hmc(
        logp,
        grad,
        approx.covariance,
        hmc_gvi_epsilon,
        hmc_gvi_leapfrog,
        cfg["n_samples"],
        cfg["burn_in"],
        rng,
        init=hmc_gvi_init,
    )
    hmc_gvi_main_time = now() - hmc_gvi_main_start
    rows.append(
        {
            "method": "HMC-GVI",
            "time_sec": gvi_time + hmc_gvi_tune_time + hmc_gvi_main_time,
            "preprocess_time_sec": gvi_time + hmc_gvi_tune_time,
            "sampling_time_sec": hmc_gvi_main_time,
            "covariance_samples": np.nan,
            "covariance_source": gvi_source,
            "burn_covariance_time_sec": np.nan,
            "gvi_covariance_time_sec": gvi_time,
            "efficiency": efficiency(hmc_gvi.samples),
            "p_jump": hmc_gvi.acceptance_rate,
            "rho1": lag1_autocorrelation(hmc_gvi.samples),
            "epsilon": hmc_gvi_epsilon,
            "L": hmc_gvi_leapfrog,
        }
    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results" / "glmm.csv"),
    )
    parser.add_argument("--tuning-profile", choices=("paper", "retuned"), default="paper")
    args = parser.parse_args()

    cfg = settings_for_profile(args.tuning_profile)
    df = run(seed=args.seed, cfg=cfg)
    df["tuning_profile"] = args.tuning_profile
    df = add_derived_mcmc_metrics(df, int(cfg["n_samples"]))
    output_path = Path(args.output)
    print(df.to_string(index=False))
    saved_to = safe_to_csv(df, output_path)
    print(json.dumps({"saved_to": str(saved_to)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
