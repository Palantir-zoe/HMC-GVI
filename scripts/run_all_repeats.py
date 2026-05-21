from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import load_german_credit_dataset, load_pima_dataset, load_polypharm_dataset
from mcmc import adaptive_mh, hmc, mala, random_walk_mh
from metrics import efficiency, lag1_autocorrelation, mean_ess
from targets import (
    logistic_grad_logposterior,
    logistic_logposterior,
    polypharm_grad_logposterior,
    polypharm_logposterior,
)
from tuning import tune_hmc_hyperparameters, tune_mala_epsilon
from utils import draw_gaussian_covariance, now, regularized_sample_covariance, set_seed
from vi import cgvi, fit_fgvi_to_gaussian, sparse_precision_gvi_polypharm


ROOT = Path(__file__).resolve().parents[1]
METHOD_OFFSETS = {
    "RMH": 11,
    "AM": 23,
    "MALA": 37,
    "MALA-GVI": 41,
    "HMC": 53,
    "HMC-GVI": 67,
}


SETTINGS = {
    "n_samples": 1_000_000,
    "burn_in": 10_000,
    "pre_burn": 10_000,
    "burn_cov_pre_burn": 10_000,
    "glmm_mala_burn_cov_pre_burn": 100_000,
    "mala_tune_rounds": 8,
    "mala_tune_steps": 200,
    "hmc_tune_rounds": 8,
    "hmc_tune_steps": 180,
    "hmc_eval_steps": 500,
    "hmc_leapfrog_candidates": (4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64),
    "pima_hmc_gvi_leapfrog_candidates": (4, 6, 8, 10, 12, 16, 20, 24),
    "gaussian_fgvi_rank": 5,
    "gaussian_fgvi_max_iter": 7_500,
    "gaussian_fgvi_n_samples": 150,
    "glmm_gvi_max_iter": 20_000,
    "glmm_gvi_lb_window": 10_000,
    "glmm_gvi_max_patience": 50,
    "glmm_hmc_burn_leapfrog_candidates": (16, 20, 32, 40),
    "glmm_hmc_main_leapfrog_candidates": (8, 10, 12, 16, 20, 24, 32, 40),
    "glmm_hmc_gvi_leapfrog_candidates": (8, 10, 12, 16, 20, 24, 32, 40),
}


RESULT_COLUMNS = [
    "experiment",
    "dataset",
    "repeat",
    "method",
    "seed",
    "target_seed",
    "n_samples",
    "burn_in",
    "pre_burn",
    "tuning_profile",
    "time_sec",
    "preprocess_time_sec",
    "sampling_time_sec",
    "covariance_samples",
    "covariance_source",
    "burn_covariance_time_sec",
    "gvi_covariance_time_sec",
    "tuning_time_sec",
    "ess_mean",
    "efficiency",
    "p_jump",
    "rho1",
    "epsilon",
    "L",
    "hmc_initial_epsilon",
    "hmc_initial_L",
    "hmc_gvi_tune_acceptance",
    "hmc_gvi_tune_efficiency",
    "hmc_gvi_tune_rho1",
    "gvi_iterations",
    "gvi_rank",
]


def log_event(log_path: Path, event: str, **payload: object) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, **payload}
    text = json.dumps(record, ensure_ascii=False)
    print(text, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def append_row(output: Path, row: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = {column: row.get(column, np.nan) for column in RESULT_COLUMNS}
    frame = pd.DataFrame([normalized], columns=RESULT_COLUMNS)
    frame.to_csv(output, mode="a", header=not output.exists(), index=False)


def completed_methods(output: Path) -> set[tuple[str, int, str]]:
    if not output.exists() or output.stat().st_size == 0:
        return set()
    try:
        frame = pd.read_csv(output)
    except Exception:
        return set()
    required = {"dataset", "repeat", "method"}
    if not required.issubset(frame.columns):
        return set()
    complete = set()
    for _, row in frame.dropna(subset=["dataset", "repeat", "method"]).iterrows():
        complete.add((str(row["dataset"]), int(row["repeat"]), str(row["method"])))
    return complete


def evaluate_samples(samples: np.ndarray) -> tuple[float, float, float]:
    ess = mean_ess(samples)
    eff = efficiency(samples)
    rho1 = lag1_autocorrelation(samples)
    return ess, eff, rho1


def base_row(
    *,
    experiment: str,
    dataset: str,
    repeat: int,
    method: str,
    seed: int,
    cfg: dict,
) -> dict[str, object]:
    pre_burn = int(cfg["pre_burn"])
    if method in {"MALA", "HMC"}:
        pre_burn = int(cfg.get("burn_cov_pre_burn", pre_burn))
    return {
        "experiment": experiment,
        "dataset": dataset,
        "repeat": repeat,
        "method": method,
        "seed": seed,
        "n_samples": int(cfg["n_samples"]),
        "burn_in": int(cfg["burn_in"]),
        "pre_burn": pre_burn,
        "tuning_profile": str(cfg.get("tuning_profile", "expanded_hmc_gvi_tuned")),
    }


def finish_row(
    row: dict[str, object],
    *,
    total_time: float,
    preprocess_time: float,
    sampling_time: float,
    covariance_samples: int | float,
    covariance_source: str,
    burn_covariance_time: float | None,
    gvi_covariance_time: float | None,
    samples: np.ndarray,
    acceptance_rate: float,
    epsilon: float | None,
    n_leapfrog: int | None,
    tuning_time: float,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    ess, eff, rho1 = evaluate_samples(samples)
    row.update(
        {
            "time_sec": total_time,
            "preprocess_time_sec": preprocess_time,
            "sampling_time_sec": sampling_time,
            "covariance_samples": covariance_samples,
            "covariance_source": covariance_source,
            "burn_covariance_time_sec": np.nan if burn_covariance_time is None else burn_covariance_time,
            "gvi_covariance_time_sec": np.nan if gvi_covariance_time is None else gvi_covariance_time,
            "tuning_time_sec": tuning_time,
            "ess_mean": ess,
            "efficiency": eff,
            "p_jump": acceptance_rate,
            "rho1": rho1,
            "epsilon": np.nan if epsilon is None else epsilon,
            "L": np.nan if n_leapfrog is None else n_leapfrog,
        }
    )
    if extra:
        row.update(extra)
    return row


def method_seed(base_seed: int, repeat: int, method: str, dataset_offset: int) -> int:
    return int(base_seed + 10_000 * repeat + dataset_offset + METHOD_OFFSETS[method])


def run_rmh(
    *,
    logp: Callable[[np.ndarray], float],
    dim: int,
    cfg: dict,
    rng: np.random.Generator,
) -> tuple[dict[str, object], np.ndarray, float]:
    start = now()
    cov_start = now()
    burn = random_walk_mh(logp, np.eye(dim) * 1e-3, int(cfg["pre_burn"]), 0, rng)
    proposal_cov = 2.38**2 * regularized_sample_covariance(burn.samples, np.eye(dim), shrinkage=0.0) / dim
    cov_time = now() - cov_start
    sample_start = now()
    result = random_walk_mh(logp, proposal_cov, int(cfg["n_samples"]), int(cfg["burn_in"]), rng)
    sample_time = now() - sample_start
    payload = {
        "total_time": now() - start,
        "preprocess_time": cov_time,
        "sampling_time": sample_time,
        "covariance_samples": int(burn.samples.shape[0]),
        "covariance_source": "rmh_pre_burn",
        "burn_covariance_time": cov_time,
        "gvi_covariance_time": None,
        "samples": result.samples,
        "acceptance_rate": result.acceptance_rate,
        "epsilon": None,
        "n_leapfrog": None,
        "tuning_time": 0.0,
    }
    return payload, result.samples, result.acceptance_rate


def run_am(
    *,
    logp: Callable[[np.ndarray], float],
    dim: int,
    cfg: dict,
    rng: np.random.Generator,
) -> dict[str, object]:
    start = now()
    result = adaptive_mh(logp, dim, int(cfg["n_samples"]), int(cfg["burn_in"]), rng)
    runtime = now() - start
    return {
        "total_time": runtime,
        "preprocess_time": 0.0,
        "sampling_time": runtime,
        "covariance_samples": np.nan,
        "covariance_source": "adaptive",
        "burn_covariance_time": None,
        "gvi_covariance_time": None,
        "samples": result.samples,
        "acceptance_rate": result.acceptance_rate,
        "epsilon": None,
        "n_leapfrog": None,
        "tuning_time": 0.0,
    }


def run_mala_with_burn_cov(
    *,
    logp: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    dim: int,
    cfg: dict,
    rng: np.random.Generator,
    initial_epsilon: float,
    epsilon_bounds: tuple[float, float] = (1e-3, 15.0),
) -> dict[str, object]:
    start = now()
    cov_start = now()
    burn_cov_pre_burn = int(cfg.get("burn_cov_pre_burn", cfg["pre_burn"]))
    tune1 = tune_mala_epsilon(
        logp,
        grad,
        np.eye(dim),
        rng,
        initial_epsilon=initial_epsilon,
        tuning_rounds=int(cfg["mala_tune_rounds"]),
        tuning_steps=int(cfg["mala_tune_steps"]),
        epsilon_bounds=epsilon_bounds,
    )
    burn = mala(logp, grad, np.eye(dim), tune1.epsilon, burn_cov_pre_burn, 0, rng, init=tune1.final_state)
    cov = regularized_sample_covariance(burn.samples, np.eye(dim), shrinkage=0.0)
    cov_time = now() - cov_start
    covariance_samples = int(burn.samples.shape[0])
    burn_final = burn.samples[-1].copy()
    tune2_start = now()
    tune2 = tune_mala_epsilon(
        logp,
        grad,
        cov,
        rng,
        init=burn_final,
        initial_epsilon=tune1.epsilon,
        tuning_rounds=int(cfg["mala_tune_rounds"]),
        tuning_steps=int(cfg["mala_tune_steps"]),
        epsilon_bounds=epsilon_bounds,
    )
    tune2_time = now() - tune2_start
    del burn
    gc.collect()
    sample_start = now()
    result = mala(
        logp,
        grad,
        cov,
        tune2.epsilon,
        int(cfg["n_samples"]),
        int(cfg["burn_in"]),
        rng,
        init=tune2.final_state,
    )
    sample_time = now() - sample_start
    return {
        "total_time": now() - start,
        "preprocess_time": cov_time + tune2_time,
        "sampling_time": sample_time,
        "covariance_samples": covariance_samples,
        "covariance_source": "mala_pre_burn",
        "burn_covariance_time": cov_time,
        "gvi_covariance_time": None,
        "samples": result.samples,
        "acceptance_rate": result.acceptance_rate,
        "epsilon": tune2.epsilon,
        "n_leapfrog": None,
        "tuning_time": tune2_time,
    }


def tune_hmc(
    *,
    logp: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    mass: np.ndarray,
    rng: np.random.Generator,
    cfg: dict,
    init: np.ndarray | None,
    initial_epsilon: float,
    leapfrog_candidates: Iterable[int],
    epsilon_bounds: tuple[float, float],
) -> tuple[object, float]:
    start = now()
    tuned = tune_hmc_hyperparameters(
        logp,
        grad,
        mass,
        rng,
        init=init,
        initial_epsilon=initial_epsilon,
        leapfrog_candidates=tuple(leapfrog_candidates),
        target_acceptance=0.95,
        min_acceptance=0.80,
        tuning_rounds=int(cfg["hmc_tune_rounds"]),
        tuning_steps=int(cfg["hmc_tune_steps"]),
        evaluation_steps=int(cfg["hmc_eval_steps"]),
        epsilon_bounds=epsilon_bounds,
    )
    return tuned, now() - start


def run_hmc_with_burn_cov(
    *,
    logp: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    dim: int,
    cfg: dict,
    rng: np.random.Generator,
    initial_epsilon: float,
    leapfrog_candidates: Iterable[int],
    epsilon_bounds: tuple[float, float],
) -> dict[str, object]:
    start = now()
    burn_cov_pre_burn = int(cfg.get("burn_cov_pre_burn", cfg["pre_burn"]))
    tune1, tune1_time = tune_hmc(
        logp=logp,
        grad=grad,
        mass=np.eye(dim),
        rng=rng,
        cfg=cfg,
        init=None,
        initial_epsilon=initial_epsilon,
        leapfrog_candidates=leapfrog_candidates,
        epsilon_bounds=epsilon_bounds,
    )
    pre_start = now()
    burn = hmc(
        logp,
        grad,
        np.eye(dim),
        tune1.epsilon,
        int(tune1.n_leapfrog),
        burn_cov_pre_burn,
        0,
        rng,
        init=tune1.final_state,
    )
    pre_time = now() - pre_start
    cov_start = now()
    cov = regularized_sample_covariance(burn.samples, np.eye(dim), shrinkage=0.0)
    cov_time = now() - cov_start
    burn_cov_time = tune1_time + pre_time + cov_time
    covariance_samples = int(burn.samples.shape[0])
    burn_final = burn.samples[-1].copy()
    tune2, tune2_time = tune_hmc(
        logp=logp,
        grad=grad,
        mass=cov,
        rng=rng,
        cfg=cfg,
        init=burn_final,
        initial_epsilon=tune1.epsilon,
        leapfrog_candidates=leapfrog_candidates,
        epsilon_bounds=epsilon_bounds,
    )
    del burn
    gc.collect()
    sample_start = now()
    result = hmc(
        logp,
        grad,
        cov,
        tune2.epsilon,
        int(tune2.n_leapfrog),
        int(cfg["n_samples"]),
        int(cfg["burn_in"]),
        rng,
        init=tune2.final_state,
    )
    sample_time = now() - sample_start
    return {
        "total_time": now() - start,
        "preprocess_time": burn_cov_time + tune2_time,
        "sampling_time": sample_time,
        "covariance_samples": covariance_samples,
        "covariance_source": "hmc_pre_burn",
        "burn_covariance_time": burn_cov_time,
        "gvi_covariance_time": None,
        "samples": result.samples,
        "acceptance_rate": result.acceptance_rate,
        "epsilon": tune2.epsilon,
        "n_leapfrog": int(tune2.n_leapfrog),
        "tuning_time": tune1_time + tune2_time,
        "extra": {
            "hmc_initial_epsilon": tune1.epsilon,
            "hmc_initial_L": int(tune1.n_leapfrog),
        },
    }


def run_hmc_with_gvi_cov(
    *,
    logp: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    cov: np.ndarray,
    init: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
    gvi_time: float,
    gvi_source: str,
    initial_epsilon: float,
    leapfrog_candidates: Iterable[int],
    epsilon_bounds: tuple[float, float],
) -> dict[str, object]:
    start = now()
    tuned, tune_time = tune_hmc(
        logp=logp,
        grad=grad,
        mass=cov,
        rng=rng,
        cfg=cfg,
        init=init,
        initial_epsilon=initial_epsilon,
        leapfrog_candidates=leapfrog_candidates,
        epsilon_bounds=epsilon_bounds,
    )
    sample_start = now()
    result = hmc(
        logp,
        grad,
        cov,
        tuned.epsilon,
        int(tuned.n_leapfrog),
        int(cfg["n_samples"]),
        int(cfg["burn_in"]),
        rng,
        init=tuned.final_state,
    )
    sample_time = now() - sample_start
    return {
        "total_time": now() - start + gvi_time,
        "preprocess_time": gvi_time + tune_time,
        "sampling_time": sample_time,
        "covariance_samples": np.nan,
        "covariance_source": gvi_source,
        "burn_covariance_time": None,
        "gvi_covariance_time": gvi_time,
        "samples": result.samples,
        "acceptance_rate": result.acceptance_rate,
        "epsilon": tuned.epsilon,
        "n_leapfrog": int(tuned.n_leapfrog),
        "tuning_time": tune_time,
        "extra": {
            "hmc_gvi_tune_acceptance": tuned.acceptance_rate,
            "hmc_gvi_tune_efficiency": tuned.efficiency,
            "hmc_gvi_tune_rho1": tuned.lag1,
        },
    }


def run_mala_with_gvi_cov(
    *,
    logp: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    cov: np.ndarray,
    init: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
    gvi_time: float,
    gvi_source: str,
    initial_epsilon: float,
    epsilon_bounds: tuple[float, float],
) -> dict[str, object]:
    start = now()
    tune_start = now()
    tuned = tune_mala_epsilon(
        logp,
        grad,
        cov,
        rng,
        init=init,
        initial_epsilon=initial_epsilon,
        tuning_rounds=int(cfg["mala_tune_rounds"]),
        tuning_steps=int(cfg["mala_tune_steps"]),
        epsilon_bounds=epsilon_bounds,
    )
    tune_time = now() - tune_start
    sample_start = now()
    result = mala(
        logp,
        grad,
        cov,
        tuned.epsilon,
        int(cfg["n_samples"]),
        int(cfg["burn_in"]),
        rng,
        init=tuned.final_state,
    )
    sample_time = now() - sample_start
    return {
        "total_time": now() - start + gvi_time,
        "preprocess_time": gvi_time + tune_time,
        "sampling_time": sample_time,
        "covariance_samples": np.nan,
        "covariance_source": gvi_source,
        "burn_covariance_time": None,
        "gvi_covariance_time": gvi_time,
        "samples": result.samples,
        "acceptance_rate": result.acceptance_rate,
        "epsilon": tuned.epsilon,
        "n_leapfrog": None,
        "tuning_time": tune_time,
    }


def append_method_result(output: Path, log_path: Path, row: dict[str, object], payload: dict[str, object]) -> None:
    extra = payload.get("extra")
    complete = finish_row(row, extra=extra, **{k: v for k, v in payload.items() if k != "extra"})
    append_row(output, complete)
    log_event(
        log_path,
        "method_done",
        dataset=complete["dataset"],
        repeat=int(complete["repeat"]),
        method=complete["method"],
        time_sec=complete["time_sec"],
        efficiency=complete["efficiency"],
        ess_mean=complete["ess_mean"],
        rho1=complete["rho1"],
        p_jump=complete["p_jump"],
        epsilon=complete["epsilon"],
        L=complete["L"],
    )


def selected_methods(all_methods: Iterable[str], requested: set[str] | None) -> tuple[str, ...]:
    if requested is None:
        return tuple(all_methods)
    return tuple(method for method in all_methods if method in requested)


def run_logistic_dataset(
    *,
    dataset: str,
    y: np.ndarray,
    x: np.ndarray,
    repeat: int,
    base_seed: int,
    dataset_offset: int,
    cfg: dict,
    output: Path,
    log_path: Path,
    methods: set[str] | None = None,
) -> None:
    dim = x.shape[1]
    logp = lambda beta: logistic_logposterior(beta, y, x)
    grad = lambda beta: logistic_grad_logposterior(beta, y, x)

    for method in selected_methods(("RMH", "AM", "MALA", "HMC", "HMC-GVI"), methods):
        if (dataset, repeat, method) in completed_methods(output):
            log_event(log_path, "method_skip", dataset=dataset, repeat=repeat, method=method, reason="already_completed")
            continue
        seed = method_seed(base_seed, repeat, method, dataset_offset)
        rng = set_seed(seed)
        row = base_row(
            experiment="4.2.1_logistic",
            dataset=dataset,
            repeat=repeat,
            method=method,
            seed=seed,
            cfg=cfg,
        )
        log_event(log_path, "method_start", dataset=dataset, repeat=repeat, method=method, seed=seed)

        if method == "RMH":
            payload, _, _ = run_rmh(logp=logp, dim=dim, cfg=cfg, rng=rng)
        elif method == "AM":
            payload = run_am(logp=logp, dim=dim, cfg=cfg, rng=rng)
        elif method == "MALA":
            payload = run_mala_with_burn_cov(
                logp=logp,
                grad=grad,
                dim=dim,
                cfg=cfg,
                rng=rng,
                initial_epsilon=1.65,
            )
        elif method == "HMC":
            payload = run_hmc_with_burn_cov(
                logp=logp,
                grad=grad,
                dim=dim,
                cfg=cfg,
                rng=rng,
                initial_epsilon=0.10 if dataset == "pima" else 0.06,
                leapfrog_candidates=cfg["hmc_leapfrog_candidates"],
                epsilon_bounds=(1e-4, 1.0),
            )
        else:
            if dataset == "pima":
                row["tuning_profile"] = "pima_hmc_gvi_lcap24"
            gvi_start = now()
            approx = cgvi(logp, grad, dim=dim, rng=rng, max_iter=8_000, n_samples=150)
            gvi_time = now() - gvi_start
            payload = run_hmc_with_gvi_cov(
                logp=logp,
                grad=grad,
                cov=approx.covariance,
                init=approx.mean,
                cfg=cfg,
                rng=rng,
                gvi_time=gvi_time,
                gvi_source="cgvi",
                initial_epsilon=0.16 if dataset == "pima" else 0.12,
                leapfrog_candidates=cfg["pima_hmc_gvi_leapfrog_candidates"]
                if dataset == "pima"
                else cfg["hmc_leapfrog_candidates"],
                epsilon_bounds=(1e-4, 1.5),
            )
            payload.setdefault("extra", {})
            payload["extra"].update({"gvi_iterations": approx.metadata.get("iterations")})

        append_method_result(output, log_path, row, payload)


def run_gaussian_100d(
    *,
    repeat: int,
    base_seed: int,
    cfg: dict,
    output: Path,
    log_path: Path,
    methods: set[str] | None = None,
) -> None:
    target_seed = base_seed + 20_000 * repeat + 3_000
    target_rng = set_seed(target_seed)
    dim = 100
    mean = np.zeros(dim, dtype=float)
    cov = draw_gaussian_covariance(dim, target_rng)
    precision = np.linalg.inv(cov)
    logp = lambda theta: float(-0.5 * (theta - mean) @ precision @ (theta - mean))
    grad = lambda theta: -(precision @ (theta - mean))

    for method in selected_methods(("RMH", "AM", "MALA", "HMC", "HMC-GVI"), methods):
        if ("gaussian_100d", repeat, method) in completed_methods(output):
            log_event(
                log_path,
                "method_skip",
                dataset="gaussian_100d",
                repeat=repeat,
                method=method,
                reason="already_completed",
            )
            continue
        seed = method_seed(base_seed, repeat, method, 3_000)
        rng = set_seed(seed)
        row = base_row(
            experiment="4.2.2_gaussian_100d",
            dataset="gaussian_100d",
            repeat=repeat,
            method=method,
            seed=seed,
            cfg=cfg,
        )
        row["target_seed"] = target_seed
        log_event(log_path, "method_start", dataset="gaussian_100d", repeat=repeat, method=method, seed=seed)

        if method == "RMH":
            payload, _, _ = run_rmh(logp=logp, dim=dim, cfg=cfg, rng=rng)
        elif method == "AM":
            payload = run_am(logp=logp, dim=dim, cfg=cfg, rng=rng)
        elif method == "MALA":
            payload = run_mala_with_burn_cov(
                logp=logp,
                grad=grad,
                dim=dim,
                cfg=cfg,
                rng=rng,
                initial_epsilon=1.65,
            )
        elif method == "HMC":
            payload = run_hmc_with_burn_cov(
                logp=logp,
                grad=grad,
                dim=dim,
                cfg=cfg,
                rng=rng,
                initial_epsilon=0.16,
                leapfrog_candidates=cfg["hmc_leapfrog_candidates"],
                epsilon_bounds=(1e-4, 1.0),
            )
        else:
            row["tuning_profile"] = "hmc_gvi_grid_rho_constrained"
            gvi_start = now()
            approx = fit_fgvi_to_gaussian(
                mean,
                cov,
                rank=int(cfg["gaussian_fgvi_rank"]),
                seed=seed,
                max_iter=int(cfg["gaussian_fgvi_max_iter"]),
                n_samples=int(cfg["gaussian_fgvi_n_samples"]),
            )
            gvi_time = now() - gvi_start
            payload = run_hmc_with_gvi_cov(
                logp=logp,
                grad=grad,
                cov=approx.covariance,
                init=approx.mean,
                cfg=cfg,
                rng=rng,
                gvi_time=gvi_time,
                gvi_source="fgvi_rank5",
                initial_epsilon=0.20,
                leapfrog_candidates=cfg["hmc_leapfrog_candidates"],
                epsilon_bounds=(1e-4, 1.5),
            )
            payload.setdefault("extra", {})
            payload["extra"].update(
                {
                    "gvi_iterations": approx.metadata.get("iterations"),
                    "gvi_rank": approx.metadata.get("rank"),
                }
            )

        append_method_result(output, log_path, row, payload)


def run_glmm(
    *,
    repeat: int,
    base_seed: int,
    cfg: dict,
    output: Path,
    log_path: Path,
    methods: set[str] | None = None,
) -> None:
    data = load_polypharm_dataset()
    dim = data["n_subjects"] + 8 + 1
    logp = lambda theta: polypharm_logposterior(theta, data)
    grad = lambda theta: polypharm_grad_logposterior(theta, data)

    gvi_cache: tuple[object, float] | None = None

    for method in selected_methods(("MALA", "MALA-GVI", "HMC", "HMC-GVI"), methods):
        if ("glmm_polypharmacy", repeat, method) in completed_methods(output):
            log_event(
                log_path,
                "method_skip",
                dataset="glmm_polypharmacy",
                repeat=repeat,
                method=method,
                reason="already_completed",
            )
            continue
        seed = method_seed(base_seed, repeat, method, 4_000)
        rng = set_seed(seed)
        method_cfg = cfg
        if method == "MALA":
            method_cfg = {
                **cfg,
                "burn_cov_pre_burn": int(cfg["glmm_mala_burn_cov_pre_burn"]),
                "tuning_profile": f"expanded_hmc_gvi_tuned_burncov{int(cfg['glmm_mala_burn_cov_pre_burn'])}",
            }
        elif method == "MALA-GVI":
            method_cfg = {
                **cfg,
                "tuning_profile": f"expanded_hmc_gvi_tuned_burncov{int(cfg['glmm_mala_burn_cov_pre_burn'])}",
            }
        elif method == "HMC-GVI":
            method_cfg = {**cfg, "tuning_profile": "hmc_gvi_grid_rho_constrained"}
        row = base_row(
            experiment="4.2.3_glmm",
            dataset="glmm_polypharmacy",
            repeat=repeat,
            method=method,
            seed=seed,
            cfg=method_cfg,
        )
        log_event(log_path, "method_start", dataset="glmm_polypharmacy", repeat=repeat, method=method, seed=seed)

        if method == "MALA":
            payload = run_mala_with_burn_cov(
                logp=logp,
                grad=grad,
                dim=dim,
                cfg={**method_cfg, "mala_tune_steps": 150},
                rng=rng,
                initial_epsilon=0.15,
                epsilon_bounds=(1e-3, 1.0),
            )
        elif method == "HMC":
            payload = run_hmc_with_burn_cov(
                logp=logp,
                grad=grad,
                dim=dim,
                cfg={**cfg, "hmc_tune_steps": 100, "hmc_eval_steps": 260},
                rng=rng,
                initial_epsilon=0.03,
                leapfrog_candidates=cfg["glmm_hmc_main_leapfrog_candidates"],
                epsilon_bounds=(1e-4, 0.15),
            )
        else:
            if gvi_cache is None:
                gvi_start = now()
                approx = sparse_precision_gvi_polypharm(
                    logp,
                    grad,
                    dim=dim,
                    random_effect_dim=data["n_subjects"],
                    rng=rng,
                    max_iter=int(cfg["glmm_gvi_max_iter"]),
                    lb_window=int(cfg["glmm_gvi_lb_window"]),
                    max_patience=int(cfg["glmm_gvi_max_patience"]),
                )
                gvi_cache = (approx, now() - gvi_start)
            approx, gvi_time = gvi_cache
            if method == "MALA-GVI":
                payload = run_mala_with_gvi_cov(
                    logp=logp,
                    grad=grad,
                    cov=approx.covariance,
                    init=approx.mean,
                    cfg={**method_cfg, "mala_tune_steps": 150},
                    rng=rng,
                    gvi_time=gvi_time,
                    gvi_source="sparse_precision_gvi",
                    initial_epsilon=0.5,
                    epsilon_bounds=(1e-3, 1.2),
                )
            else:
                payload = run_hmc_with_gvi_cov(
                    logp=logp,
                    grad=grad,
                    cov=approx.covariance,
                    init=approx.mean,
                    cfg={**method_cfg, "hmc_tune_steps": 100, "hmc_eval_steps": 260},
                    rng=rng,
                    gvi_time=gvi_time,
                    gvi_source="sparse_precision_gvi",
                    initial_epsilon=0.16,
                    leapfrog_candidates=cfg["glmm_hmc_gvi_leapfrog_candidates"],
                    epsilon_bounds=(1e-4, 0.30),
                )
            payload.setdefault("extra", {})
            payload["extra"].update({"gvi_iterations": approx.metadata.get("iterations")})

        append_method_result(output, log_path, row, payload)


def write_summary(output: Path, summary_output: Path) -> None:
    if not output.exists():
        return
    frame = pd.read_csv(output)
    summary = (
        frame.groupby(["experiment", "dataset", "method"], dropna=False)
        .agg(
            repeats=("repeat", "count"),
            time_sec_mean=("time_sec", "mean"),
            sampling_time_sec_mean=("sampling_time_sec", "mean"),
            burn_covariance_time_sec_mean=("burn_covariance_time_sec", "mean"),
            gvi_covariance_time_sec_mean=("gvi_covariance_time_sec", "mean"),
            ess_mean_mean=("ess_mean", "mean"),
            efficiency_mean=("efficiency", "mean"),
            efficiency_sd=("efficiency", "std"),
            p_jump_mean=("p_jump", "mean"),
            rho1_mean=("rho1", "mean"),
            epsilon_mean=("epsilon", "mean"),
            L_mean=("L", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_output, index=False)


def parse_targets(text: str) -> list[str]:
    allowed = ["pima", "german", "gaussian_100d", "glmm"]
    if text == "all":
        return allowed
    targets = [part.strip() for part in text.split(",") if part.strip()]
    unknown = sorted(set(targets) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}")
    return targets


def parse_methods(text: str) -> set[str] | None:
    allowed = {"RMH", "AM", "MALA", "MALA-GVI", "HMC", "HMC-GVI"}
    if text.lower() == "all":
        return None
    methods = {part.strip().upper() for part in text.split(",") if part.strip()}
    unknown = sorted(methods - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=str(ROOT / "results" / "all_repeats.csv"))
    parser.add_argument("--summary-output", type=str, default=str(ROOT / "results" / "all_repeats_summary.csv"))
    parser.add_argument("--log", type=str, default=str(ROOT / "results" / "logs" / "all_repeats_progress.log"))
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--targets", type=str, default="all")
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--burn-cov-pre-burn", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    summary_output = Path(args.summary_output)
    log_path = Path(args.log)
    if not args.resume:
        for path in (output, summary_output, log_path):
            if path.exists():
                path.unlink()

    cfg = dict(SETTINGS)
    if args.burn_cov_pre_burn is not None:
        cfg["burn_cov_pre_burn"] = int(args.burn_cov_pre_burn)
    targets = parse_targets(args.targets)
    methods = parse_methods(args.methods)
    log_event(
        log_path,
        "start",
        targets=targets,
        methods="all" if methods is None else sorted(methods),
        repeats=args.repeats,
        settings=cfg,
        output=str(output),
    )

    pima = None
    german = None
    for target in targets:
        for repeat in range(1, args.repeats + 1):
            if target == "pima":
                if pima is None:
                    pima = load_pima_dataset()
                y, x = pima
                run_logistic_dataset(
                    dataset="pima",
                    y=y,
                    x=x,
                    repeat=repeat,
                    base_seed=args.base_seed,
                    dataset_offset=1_000,
                    cfg=cfg,
                    output=output,
                    log_path=log_path,
                    methods=methods,
                )
                write_summary(output, summary_output)

            elif target == "german":
                if german is None:
                    german = load_german_credit_dataset()
                y, x = german
                run_logistic_dataset(
                    dataset="german",
                    y=y,
                    x=x,
                    repeat=repeat,
                    base_seed=args.base_seed,
                    dataset_offset=2_000,
                    cfg=cfg,
                    output=output,
                    log_path=log_path,
                    methods=methods,
                )
                write_summary(output, summary_output)

            elif target == "gaussian_100d":
                run_gaussian_100d(
                    repeat=repeat,
                    base_seed=args.base_seed,
                    cfg=cfg,
                    output=output,
                    log_path=log_path,
                    methods=methods,
                )
                write_summary(output, summary_output)

            elif target == "glmm":
                run_glmm(
                    repeat=repeat,
                    base_seed=args.base_seed,
                    cfg=cfg,
                    output=output,
                    log_path=log_path,
                    methods=methods,
                )
                write_summary(output, summary_output)

    write_summary(output, summary_output)
    log_event(log_path, "done", output=str(output), summary_output=str(summary_output))


if __name__ == "__main__":
    main()
