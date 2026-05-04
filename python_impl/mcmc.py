from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .utils import ensure_spd


Array = np.ndarray


@dataclass
class SamplerResult:
    samples: Array
    acceptance_rate: float
    accepted: int
    total: int


def random_walk_mh(
    log_density: Callable[[Array], float],
    proposal_cov: Array,
    n_samples: int,
    burn_in: int,
    rng: np.random.Generator,
    init: Array | None = None,
) -> SamplerResult:
    proposal_cov = ensure_spd(proposal_cov)
    dim = proposal_cov.shape[0]
    chol = np.linalg.cholesky(proposal_cov)
    x = np.zeros(dim, dtype=float) if init is None else np.array(init, dtype=float)
    logp = log_density(x)
    samples = np.zeros((n_samples, dim), dtype=float)
    accepted = 0
    total = 0

    for step in range(n_samples + burn_in):
        proposal = x + chol @ rng.normal(size=dim)
        logp_new = log_density(proposal)
        total += 1
        if np.log(rng.uniform()) < logp_new - logp:
            x = proposal
            logp = logp_new
            accepted += 1
        if step >= burn_in:
            samples[step - burn_in] = x

    return SamplerResult(samples=samples, acceptance_rate=accepted / total, accepted=accepted, total=total)


def adaptive_mh(
    log_density: Callable[[Array], float],
    dim: int,
    n_samples: int,
    burn_in: int,
    rng: np.random.Generator,
    epsilon: float = 1e-6,
    init: Array | None = None,
) -> SamplerResult:
    x = np.zeros(dim, dtype=float) if init is None else np.array(init, dtype=float)
    logp = log_density(x)
    mu = x.copy()
    sigma = np.zeros((dim, dim), dtype=float)
    samples = np.zeros((n_samples, dim), dtype=float)
    accepted = 0
    total = 0

    for step in range(1, n_samples + burn_in + 1):
        if step == 1:
            proposal_cov = np.eye(dim) * epsilon
        else:
            proposal_cov = 2.38**2 * sigma / dim + np.eye(dim) * epsilon
            proposal_cov = ensure_spd(proposal_cov)
        chol = np.linalg.cholesky(proposal_cov)
        proposal = x + chol @ rng.normal(size=dim)
        logp_new = log_density(proposal)
        total += 1
        if np.log(rng.uniform()) < logp_new - logp:
            x = proposal
            logp = logp_new
            accepted += 1
        if step > 1:
            delta = x - mu
            sigma = (step - 2) / (step - 1) * sigma + np.outer(delta, delta) / step
            mu = (step - 1) / step * mu + x / step
        if step > burn_in:
            samples[step - burn_in - 1] = x

    return SamplerResult(samples=samples, acceptance_rate=accepted / total, accepted=accepted, total=total)


def metropolis_within_gibbs_gaussian(
    mean: Array,
    covariance: Array,
    step_size: float,
    time_budget_seconds: float,
    rng: np.random.Generator,
    init: Array | None = None,
    min_iterations: int = 1,
) -> SamplerResult:
    precision = np.linalg.inv(covariance)
    dim = covariance.shape[0]
    x = np.zeros(dim, dtype=float) if init is None else np.array(init, dtype=float)

    def log_density(theta: Array) -> float:
        delta = theta - mean
        return float(-0.5 * delta @ precision @ delta)

    import time

    started = time.perf_counter()
    rows = []
    accepted = 0
    total = 0
    current_logp = log_density(x)

    while time.perf_counter() - started < time_budget_seconds or len(rows) < min_iterations:
        proposal = x.copy()
        for j in range(dim):
            proposal[j] = x[j] + rng.normal(scale=step_size)
            proposal_logp = log_density(proposal)
            total += 1
            if np.log(rng.uniform()) < proposal_logp - current_logp:
                x[j] = proposal[j]
                current_logp = proposal_logp
                accepted += 1
            else:
                proposal[j] = x[j]
        rows.append(x.copy())

    samples = np.vstack(rows) if rows else x.reshape(1, -1)
    return SamplerResult(samples=samples, acceptance_rate=accepted / max(total, 1), accepted=accepted, total=total)


def mala(
    log_density: Callable[[Array], float],
    grad_log_density: Callable[[Array], Array],
    preconditioner: Array,
    epsilon: float,
    n_samples: int,
    burn_in: int,
    rng: np.random.Generator,
    init: Array | None = None,
) -> SamplerResult:
    preconditioner = ensure_spd(preconditioner)
    dim = preconditioner.shape[0]
    chol = np.linalg.cholesky(preconditioner)
    inv_preconditioner = np.linalg.inv(preconditioner)
    x = np.zeros(dim, dtype=float) if init is None else np.array(init, dtype=float)
    grad = preconditioner @ grad_log_density(x)
    logp = log_density(x)
    sigma2 = epsilon**2 / dim ** (1.0 / 3.0)
    sigma = np.sqrt(sigma2)
    samples = np.zeros((n_samples, dim), dtype=float)
    accepted = 0
    total = 0

    for step in range(n_samples + burn_in):
        proposal = x + 0.5 * sigma2 * grad + sigma * (chol.T @ rng.normal(size=dim))
        logp_new = log_density(proposal)
        grad_new = preconditioner @ grad_log_density(proposal)

        diff_old = x - proposal - 0.5 * sigma2 * grad_new
        diff_new = proposal - x - 0.5 * sigma2 * grad
        q_old = -0.5 / sigma2 * diff_old @ inv_preconditioner @ diff_old
        q_new = -0.5 / sigma2 * diff_new @ inv_preconditioner @ diff_new

        total += 1
        if np.log(rng.uniform()) < logp_new - logp + q_old - q_new:
            x = proposal
            logp = logp_new
            grad = grad_new
            accepted += 1
        if step >= burn_in:
            samples[step - burn_in] = x

    return SamplerResult(samples=samples, acceptance_rate=accepted / total, accepted=accepted, total=total)


def hmc(
    log_density: Callable[[Array], float],
    grad_log_density: Callable[[Array], Array],
    mass_matrix: Array,
    epsilon: float,
    n_leapfrog: int,
    n_samples: int,
    burn_in: int,
    rng: np.random.Generator,
    init: Array | None = None,
) -> SamplerResult:
    mass_matrix = ensure_spd(mass_matrix)
    dim = mass_matrix.shape[0]
    chol_mass = np.linalg.cholesky(mass_matrix)
    x = np.zeros(dim, dtype=float) if init is None else np.array(init, dtype=float)
    logp = log_density(x)
    samples = np.zeros((n_samples, dim), dtype=float)
    accepted = 0
    total = 0

    for step in range(n_samples + burn_in):
        momentum = np.linalg.solve(chol_mass.T, rng.normal(size=dim))
        logk = 0.5 * momentum @ mass_matrix @ momentum

        x_new = x.copy()
        p_new = momentum + 0.5 * epsilon * grad_log_density(x_new)
        for leap in range(n_leapfrog):
            x_new = x_new + epsilon * (mass_matrix @ p_new)
            if leap != n_leapfrog - 1:
                p_new = p_new + epsilon * grad_log_density(x_new)
        p_new = p_new + 0.5 * epsilon * grad_log_density(x_new)
        p_new = -p_new

        logp_new = log_density(x_new)
        logk_new = 0.5 * p_new @ mass_matrix @ p_new
        total += 1
        if np.log(rng.uniform()) < logp_new - logp + logk - logk_new:
            x = x_new
            logp = logp_new
            accepted += 1
        if step >= burn_in:
            samples[step - burn_in] = x

    return SamplerResult(samples=samples, acceptance_rate=accepted / total, accepted=accepted, total=total)

