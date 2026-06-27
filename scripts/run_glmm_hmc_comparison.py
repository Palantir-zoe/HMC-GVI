from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import load_polypharm_dataset
from mcmc import hmc
from scripts.run_all_repeats import (
    SETTINGS,
    append_method_result,
    base_row,
    completed_methods,
    log_event,
    method_seed,
    run_glmm,
    write_summary,
)
from targets import polypharm_grad_logposterior, polypharm_logposterior
from utils import now, set_seed
from vi import sparse_precision_gvi_polypharm


def parse_methods(text: str) -> set[str]:
    allowed = {"HMC", "HMC-GVI"}
    if text.lower() == "all":
        return allowed
    methods = {part.strip().upper() for part in text.split(",") if part.strip()}
    unknown = sorted(methods - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return methods


def run_pilot_selected_hmc_gvi(
    *,
    repeat: int,
    base_seed: int,
    cfg: dict,
    output: Path,
    log_path: Path,
    epsilon: float,
    leapfrog: int,
) -> None:
    if ("glmm_polypharmacy", repeat, "HMC-GVI") in completed_methods(output):
        log_event(log_path, "method_skip", dataset="glmm_polypharmacy", repeat=repeat, method="HMC-GVI")
        return

    data = load_polypharm_dataset()
    dim = data["n_subjects"] + 8 + 1
    logp = lambda theta: polypharm_logposterior(theta, data)
    grad = lambda theta: polypharm_grad_logposterior(theta, data)

    seed = method_seed(base_seed, repeat, "HMC-GVI", 4_000)
    rng = set_seed(seed)
    method_cfg = {**cfg, "tuning_profile": f"pilot_selected_eps{epsilon:g}_L{leapfrog}"}
    row = base_row(
        experiment="4.2.3_glmm",
        dataset="glmm_polypharmacy",
        repeat=repeat,
        method="HMC-GVI",
        seed=seed,
        cfg=method_cfg,
    )
    log_event(log_path, "method_start", dataset="glmm_polypharmacy", repeat=repeat, method="HMC-GVI", seed=seed)

    start = now()
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
    gvi_time = now() - gvi_start

    sample_start = now()
    result = hmc(
        logp,
        grad,
        approx.covariance,
        epsilon,
        leapfrog,
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
        "covariance_source": "sparse_precision_gvi",
        "burn_covariance_time": None,
        "gvi_covariance_time": gvi_time,
        "samples": result.samples,
        "acceptance_rate": result.acceptance_rate,
        "epsilon": epsilon,
        "n_leapfrog": leapfrog,
        "tuning_time": 0.0,
        "extra": {"gvi_iterations": approx.metadata.get("iterations")},
    }
    append_method_result(output, log_path, row, payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the reported GLMM HMC/HMC-GVI comparison."
    )
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "results" / "glmm_hmc_comparison.csv"))
    parser.add_argument(
        "--summary-output",
        type=str,
        default=str(REPO_ROOT / "results" / "glmm_hmc_comparison_summary.csv"),
    )
    parser.add_argument(
        "--log",
        type=str,
        default=str(REPO_ROOT / "results" / "logs" / "glmm_hmc_comparison_progress.log"),
    )
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", type=str, default="all")
    parser.add_argument("--hmc-gvi-epsilon", type=float, default=0.072)
    parser.add_argument("--hmc-gvi-leapfrog", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--pre-burn", type=int, default=None)
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

    methods = parse_methods(args.methods)
    log_event(
        log_path,
        "start",
        methods=sorted(methods),
        repeats=args.repeats,
        hmc_gvi_epsilon=args.hmc_gvi_epsilon,
        hmc_gvi_leapfrog=args.hmc_gvi_leapfrog,
        settings=cfg,
        output=str(output),
    )
    for repeat in range(1, args.repeats + 1):
        if "HMC" in methods:
            run_glmm(
                repeat=repeat,
                base_seed=args.base_seed,
                cfg=cfg,
                output=output,
                log_path=log_path,
                methods={"HMC"},
            )
        if "HMC-GVI" in methods:
            run_pilot_selected_hmc_gvi(
                repeat=repeat,
                base_seed=args.base_seed,
                cfg=cfg,
                output=output,
                log_path=log_path,
                epsilon=args.hmc_gvi_epsilon,
                leapfrog=args.hmc_gvi_leapfrog,
            )
        write_summary(output, summary_output)

    write_summary(output, summary_output)
    log_event(log_path, "done", output=str(output), summary_output=str(summary_output))
    print(json.dumps({"output": str(output), "summary_output": str(summary_output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
