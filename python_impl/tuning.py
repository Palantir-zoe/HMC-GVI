from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .mcmc import hmc, mala
from .metrics import efficiency, lag1_autocorrelation


Array = np.ndarray


@dataclass
class TuningResult:
    epsilon: float
    acceptance_rate: float
    lag1: float
    efficiency: float
    n_leapfrog: int | None = None
    final_state: Array | None = None


def _bounded_exp(log_value: float, lower: float, upper: float) -> float:
    return float(np.clip(np.exp(log_value), lower, upper))


def tune_mala_epsilon(
    log_density: Callable[[Array], float],
    grad_log_density: Callable[[Array], Array],
    preconditioner: Array,
    rng: np.random.Generator,
    init: Array | None = None,
    initial_epsilon: float = 1.0,
    target_acceptance: float = 0.574,
    tuning_rounds: int = 3,
    tuning_steps: int = 150,
    epsilon_bounds: tuple[float, float] = (1e-3, 15.0),
) -> TuningResult:
    base_state = None if init is None else np.array(init, dtype=float)
    fallback_state = np.zeros(preconditioner.shape[0], dtype=float) if base_state is None else base_state.copy()
    log_epsilon = float(np.log(np.clip(initial_epsilon, *epsilon_bounds)))
    best: TuningResult | None = None
    best_score = np.inf

    for round_idx in range(tuning_rounds):
        spread = max(0.35, 1.25 / (round_idx + 1.0))
        offsets = np.array([-1.5, -0.8, -0.3, 0.0, 0.3, 0.8, 1.5]) * spread

        for offset in offsets:
            epsilon = _bounded_exp(log_epsilon + float(offset), *epsilon_bounds)
            try:
                result = mala(
                    log_density,
                    grad_log_density,
                    preconditioner,
                    epsilon,
                    tuning_steps,
                    0,
                    rng,
                    init=base_state,
                )
            except Exception:
                continue
            acceptance = float(result.acceptance_rate)
            if not np.isfinite(acceptance) or not np.all(np.isfinite(result.samples)):
                continue
            rho1 = lag1_autocorrelation(result.samples)
            eff = efficiency(result.samples)
            score = abs(acceptance - target_acceptance) + 0.10 * abs(rho1) - 0.05 * eff

            if score < best_score:
                best_score = score
                best = TuningResult(
                    epsilon=epsilon,
                    acceptance_rate=acceptance,
                    lag1=rho1,
                    efficiency=eff,
                    final_state=result.samples[-1].copy(),
                )

        if best is None:
            return TuningResult(
                epsilon=float(np.clip(initial_epsilon, *epsilon_bounds)),
                acceptance_rate=0.0,
                lag1=1.0,
                efficiency=0.0,
                final_state=fallback_state,
            )
        base_state = best.final_state.copy()
        log_epsilon = float(np.log(np.clip(best.epsilon, *epsilon_bounds)))

    assert best is not None
    return best


def tune_hmc_hyperparameters(
    log_density: Callable[[Array], float],
    grad_log_density: Callable[[Array], Array],
    mass_matrix: Array,
    rng: np.random.Generator,
    init: Array | None = None,
    initial_epsilon: float = 0.1,
    leapfrog_candidates: Sequence[int] = (5, 10, 20),
    target_acceptance: float = 0.65,
    min_acceptance: float = 0.60,
    max_acceptance: float = 0.995,
    tuning_rounds: int = 3,
    tuning_steps: int = 80,
    evaluation_steps: int = 160,
    epsilon_bounds: tuple[float, float] = (1e-4, 2.0),
) -> TuningResult:
    base_state = None if init is None else np.array(init, dtype=float)
    fallback_state = np.zeros(mass_matrix.shape[0], dtype=float) if base_state is None else base_state.copy()
    best: TuningResult | None = None
    best_score = np.inf

    def score_result(result: TuningResult) -> float:
        acceptance = result.acceptance_rate
        if not np.isfinite(acceptance) or not np.isfinite(result.lag1) or not np.isfinite(result.efficiency):
            return np.inf
        score = abs(result.lag1) - 0.25 * min(result.efficiency, 2.0)
        if acceptance < min_acceptance:
            score += 5.0 * (min_acceptance - acceptance)
        elif acceptance > max_acceptance:
            score += 0.25 * (acceptance - max_acceptance)
        else:
            score += 0.10 * abs(acceptance - target_acceptance)
        return float(score)

    def evaluate(epsilon: float, n_leapfrog: int, n_steps: int, state: Array | None) -> TuningResult | None:
        try:
            evaluation = hmc(
                log_density,
                grad_log_density,
                mass_matrix,
                epsilon,
                int(n_leapfrog),
                n_steps,
                0,
                rng,
                init=state,
            )
        except Exception:
            return None
        acceptance = float(evaluation.acceptance_rate)
        if not np.isfinite(acceptance) or not np.all(np.isfinite(evaluation.samples)):
            return None
        return TuningResult(
            epsilon=epsilon,
            n_leapfrog=int(n_leapfrog),
            acceptance_rate=acceptance,
            lag1=lag1_autocorrelation(evaluation.samples),
            efficiency=efficiency(evaluation.samples),
            final_state=evaluation.samples[-1].copy(),
        )

    for n_leapfrog in leapfrog_candidates:
        candidate_state = None if base_state is None else base_state.copy()
        candidate_log_eps = float(np.log(np.clip(initial_epsilon, *epsilon_bounds)))
        candidate_best: TuningResult | None = None

        for round_idx in range(tuning_rounds):
            spread = max(0.20, 1.25 / (round_idx + 1.0))
            offsets = np.array([-1.6, -1.0, -0.45, 0.0, 0.45, 1.0, 1.6]) * spread
            local_best: TuningResult | None = None
            local_best_score = np.inf

            for offset in offsets:
                epsilon = _bounded_exp(candidate_log_eps + float(offset), *epsilon_bounds)
                evaluation = evaluate(epsilon, int(n_leapfrog), tuning_steps, candidate_state)
                if evaluation is None:
                    continue
                score = score_result(evaluation)

                if score < local_best_score:
                    local_best_score = score
                    local_best = evaluation

            if local_best is None:
                continue
            candidate_state = local_best.final_state.copy()
            candidate_log_eps = float(np.log(np.clip(local_best.epsilon, *epsilon_bounds)))
            candidate_best = local_best

        if candidate_best is None:
            continue
        final_evaluation = evaluate(candidate_best.epsilon, int(n_leapfrog), evaluation_steps, candidate_state)
        if final_evaluation is None:
            continue
        final_score = score_result(final_evaluation)
        if final_score < best_score:
            best_score = final_score
            best = final_evaluation

    if best is None:
        return TuningResult(
            epsilon=float(np.clip(initial_epsilon, *epsilon_bounds)),
            n_leapfrog=int(leapfrog_candidates[0]),
            acceptance_rate=0.0,
            lag1=1.0,
            efficiency=0.0,
            final_state=fallback_state,
        )
    return best
