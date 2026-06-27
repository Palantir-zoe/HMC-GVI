from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcmc import hmc
from scripts.run_all_repeats import (
    SETTINGS,
    append_method_result,
    base_row,
    completed_methods,
    log_event,
    method_seed,
    write_summary,
)
from tuning import tune_hmc_hyperparameters
from utils import draw_gaussian_covariance, now, regularized_sample_covariance, set_seed
from vi import fit_fgvi_to_gaussian


def parse_methods(text: str) -> set[str]:
    allowed = {"HMC", "HMC-GVI"}
    if text.lower() == "all":
        return allowed
    methods = {part.strip().upper() for part in text.split(",") if part.strip()}
    unknown = sorted(methods - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return methods


def run_hmc_reported_main(
    *,
    repeat: int,
    base_seed: int,
    cfg: dict,
    output: Path,
    log_path: Path,
    main_epsilon: float,
    main_leapfrog: int,
) -> None:
    if ("gaussian_100d", repeat, "HMC") in completed_methods(output):
        log_event(log_path, "method_skip", dataset="gaussian_100d", repeat=repeat, method="HMC")
        return

    target_seed = base_seed + 20_000 * repeat + 3_000
    target_rng = set_seed(target_seed)
    dim = 100
    mean = np.zeros(dim, dtype=float)
    covariance = draw_gaussian_covariance(dim, target_rng)
    precision = np.linalg.inv(covariance)
    logp = lambda theta: float(-0.5 * (theta - mean) @ precision @ (theta - mean))
    grad = lambda theta: -(precision @ (theta - mean))

    seed = method_seed(base_seed, repeat, "HMC", 3_000)
    rng = set_seed(seed)
    row = base_row(
        experiment="4.2.2_gaussian_100d",
        dataset="gaussian_100d",
        repeat=repeat,
        method="HMC",
        seed=seed,
        cfg=cfg,
    )
    row["target_seed"] = target_seed
    row["tuning_profile"] = f"matched_main_eps{main_epsilon:g}_L{main_leapfrog}"
    log_event(log_path, "method_start", dataset="gaussian_100d", repeat=repeat, method="HMC", seed=seed)

    start = now()
    tune_start = now()
    burn_tune = tune_hmc_hyperparameters(
        logp,
        grad,
        np.eye(dim),
        rng,
        initial_epsilon=0.16,
        leapfrog_candidates=cfg["hmc_leapfrog_candidates"],
        target_acceptance=0.95,
        min_acceptance=0.80,
        tuning_rounds=int(cfg["hmc_tune_rounds"]),
        tuning_steps=int(cfg["hmc_tune_steps"]),
        evaluation_steps=int(cfg["hmc_eval_steps"]),
        epsilon_bounds=(1e-4, 1.0),
    )
    tune_time = now() - tune_start

    pre_start = now()
    burn = hmc(
        logp,
        grad,
        np.eye(dim),
        burn_tune.epsilon,
        int(burn_tune.n_leapfrog),
        int(cfg["pre_burn"]),
        0,
        rng,
        init=burn_tune.final_state,
    )
    pre_time = now() - pre_start
    cov_start = now()
    mass = regularized_sample_covariance(burn.samples, np.eye(dim), shrinkage=0.0)
    cov_time = now() - cov_start

    sample_start = now()
    result = hmc(
        logp,
        grad,
        mass,
        main_epsilon,
        main_leapfrog,
        int(cfg["n_samples"]),
        int(cfg["burn_in"]),
        rng,
        init=burn.samples[-1],
    )
    sample_time = now() - sample_start

    payload = {
        "total_time": now() - start,
        "preprocess_time": tune_time + pre_time + cov_time,
        "sampling_time": sample_time,
        "covariance_samples": int(burn.samples.shape[0]),
        "covariance_source": "hmc_pre_burn",
        "burn_covariance_time": tune_time + pre_time + cov_time,
        "gvi_covariance_time": None,
        "samples": result.samples,
        "acceptance_rate": result.acceptance_rate,
        "epsilon": main_epsilon,
        "n_leapfrog": main_leapfrog,
        "tuning_time": tune_time,
        "extra": {
            "hmc_initial_epsilon": burn_tune.epsilon,
            "hmc_initial_L": int(burn_tune.n_leapfrog),
        },
    }
    append_method_result(output, log_path, row, payload)


def run_hmc_gvi_reported_main(
    *,
    repeat: int,
    base_seed: int,
    cfg: dict,
    output: Path,
    log_path: Path,
    main_epsilon: float,
    main_leapfrog: int,
) -> None:
    if ("gaussian_100d", repeat, "HMC-GVI") in completed_methods(output):
        log_event(log_path, "method_skip", dataset="gaussian_100d", repeat=repeat, method="HMC-GVI")
        return

    target_seed = base_seed + 20_000 * repeat + 3_000
    target_rng = set_seed(target_seed)
    dim = 100
    mean = np.zeros(dim, dtype=float)
    covariance = draw_gaussian_covariance(dim, target_rng)
    precision = np.linalg.inv(covariance)
    logp = lambda theta: float(-0.5 * (theta - mean) @ precision @ (theta - mean))
    grad = lambda theta: -(precision @ (theta - mean))

    seed = method_seed(base_seed, repeat, "HMC-GVI", 3_000)
    rng = set_seed(seed)
    row = base_row(
        experiment="4.2.2_gaussian_100d",
        dataset="gaussian_100d",
        repeat=repeat,
        method="HMC-GVI",
        seed=seed,
        cfg=cfg,
    )
    row["target_seed"] = target_seed
    row["tuning_profile"] = f"matched_main_eps{main_epsilon:g}_L{main_leapfrog}"
    log_event(log_path, "method_start", dataset="gaussian_100d", repeat=repeat, method="HMC-GVI", seed=seed)

    start = now()
    gvi_start = now()
    approx = fit_fgvi_to_gaussian(
        mean,
        covariance,
        rank=int(cfg["gaussian_fgvi_rank"]),
        seed=seed,
        max_iter=int(cfg["gaussian_fgvi_max_iter"]),
        n_samples=int(cfg["gaussian_fgvi_n_samples"]),
    )
    gvi_time = now() - gvi_start
    sample_start = now()
    result = hmc(
        logp,
        grad,
        approx.covariance,
        main_epsilon,
        main_leapfrog,
        int(cfg["n_samples"]),
        int(cfg["burn_in"]),
        rng,
        init=approx.mean,
    )
    sample_time = now() - sample_start

    payload = {
        "total_time": now() - start,
        "preprocess_time": gvi_time,
        "sampling_time": sample_time,
        "covariance_samples": np.nan,
        "covariance_source": "fgvi_rank5",
        "burn_covariance_time": None,
        "gvi_covariance_time": gvi_time,
        "samples": result.samples,
        "acceptance_rate": result.acceptance_rate,
        "epsilon": main_epsilon,
        "n_leapfrog": main_leapfrog,
        "tuning_time": 0.0,
        "extra": {
            "gvi_iterations": approx.metadata.get("iterations"),
            "gvi_rank": approx.metadata.get("rank"),
        },
    }
    append_method_result(output, log_path, row, payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the reported 100-dimensional Gaussian HMC/HMC-GVI comparison."
    )
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "results" / "gaussian100_matched_hmc.csv"))
    parser.add_argument(
        "--summary-output",
        type=str,
        default=str(REPO_ROOT / "results" / "gaussian100_matched_hmc_summary.csv"),
    )
    parser.add_argument(
        "--log",
        type=str,
        default=str(REPO_ROOT / "results" / "logs" / "gaussian100_matched_hmc_progress.log"),
    )
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--epsilon", type=float, default=0.16)
    parser.add_argument("--leapfrog", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--pre-burn", type=int, default=None)
    parser.add_argument("--fgvi-max-iter", type=int, default=None)
    parser.add_argument("--fgvi-n-samples", type=int, default=None)
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
    if args.n_samples is not None:
        cfg["n_samples"] = int(args.n_samples)
    if args.burn_in is not None:
        cfg["burn_in"] = int(args.burn_in)
    if args.pre_burn is not None:
        cfg["pre_burn"] = int(args.pre_burn)
    if args.fgvi_max_iter is not None:
        cfg["gaussian_fgvi_max_iter"] = int(args.fgvi_max_iter)
    if args.fgvi_n_samples is not None:
        cfg["gaussian_fgvi_n_samples"] = int(args.fgvi_n_samples)

    methods = parse_methods(args.methods)
    log_event(
        log_path,
        "start",
        methods=sorted(methods),
        repeats=args.repeats,
        epsilon=args.epsilon,
        leapfrog=args.leapfrog,
        settings=cfg,
        output=str(output),
    )
    for repeat in range(1, args.repeats + 1):
        if "HMC" in methods:
            run_hmc_reported_main(
                repeat=repeat,
                base_seed=args.base_seed,
                cfg=cfg,
                output=output,
                log_path=log_path,
                main_epsilon=args.epsilon,
                main_leapfrog=args.leapfrog,
            )
        if "HMC-GVI" in methods:
            run_hmc_gvi_reported_main(
                repeat=repeat,
                base_seed=args.base_seed,
                cfg=cfg,
                output=output,
                log_path=log_path,
                main_epsilon=args.epsilon,
                main_leapfrog=args.leapfrog,
            )
        write_summary(output, summary_output)

    write_summary(output, summary_output)
    log_event(log_path, "done", output=str(output), summary_output=str(summary_output))
    print(json.dumps({"output": str(output), "summary_output": str(summary_output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
