from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import solve_triangular

from utils import ensure_spd


@dataclass
class GaussianApproximation:
    mean: np.ndarray
    covariance: np.ndarray
    metadata: dict


def _vech_outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    matrix = np.outer(a, b)
    parts = []
    for col in range(matrix.shape[1]):
        parts.append(matrix[col:, col])
    return np.concatenate(parts)


def _unpack_cholesky_lambda(lam: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    mu = lam[:dim]
    l = np.zeros((dim, dim), dtype=float)
    index = dim
    for col in range(dim):
        count = dim - col
        l[col:, col] = lam[index : index + count]
        index += count
    return mu, l


def _regularize_triangular_factor(l: np.ndarray, min_abs_diagonal: float = 1e-8) -> np.ndarray:
    l = np.array(l, dtype=float, copy=True)
    diag = np.diag(l).copy()
    small = np.abs(diag) < min_abs_diagonal
    if np.any(small):
        diag[small] = np.where(diag[small] < 0.0, -min_abs_diagonal, min_abs_diagonal)
        np.fill_diagonal(l, diag)
    return l


def cgvi(
    log_density: Callable[[np.ndarray], float],
    grad_log_density: Callable[[np.ndarray], np.ndarray],
    dim: int,
    rng: np.random.Generator,
    n_samples: int = 100,
    max_iter: int = 5000,
    window: int = 50,
    max_patience: int = 50,
    beta1: float = 0.9,
    beta2: float = 0.9,
    e0: float = 0.01,
    tau: float = 1000.0,
) -> GaussianApproximation:
    q = dim * (dim + 1) // 2
    glen = dim + q
    lam = np.zeros((max_iter + 1, glen), dtype=float)

    cursor = dim
    for col in range(dim):
        lam[0, cursor] = 1.0
        cursor += dim - col

    lower_bounds = np.zeros(max_iter, dtype=float)
    averaged_lb = np.full(max_iter, -np.inf, dtype=float)
    gbar = np.zeros(glen, dtype=float)
    vbar = np.zeros(glen, dtype=float)

    best_index = 0
    patience = 0
    last_iter = 0

    for t in range(max_iter):
        mu, l = _unpack_cholesky_lambda(lam[t], dim)
        l = _regularize_triangular_factor(l)
        inv_l = np.linalg.solve(l, np.eye(dim))

        eps = rng.normal(size=(n_samples, dim))
        theta = mu + eps @ l.T

        grad = np.zeros(glen, dtype=float)
        lb = 0.0
        for i in range(n_samples):
            th = theta[i]
            value = (
                log_density(th)
                + np.log(np.abs(np.linalg.det(l)))
                + 0.5 * (th - mu) @ inv_l.T @ inv_l @ (th - mu)
            )
            lb += value / n_samples
            gs = grad_log_density(th) + inv_l.T @ inv_l @ (th - mu)
            grad[:dim] += gs / n_samples
            grad[dim:] += _vech_outer(gs, eps[i]) / n_samples

        lower_bounds[t] = lb
        grad_sq = grad**2
        if t == 0:
            gbar = grad
            vbar = grad_sq
        else:
            gbar = beta1 * gbar + (1.0 - beta1) * grad
            vbar = beta2 * vbar + (1.0 - beta2) * grad_sq

        if t >= window - 1:
            averaged_lb[t] = lower_bounds[t - window + 1 : t + 1].mean()
            if averaged_lb[t] >= averaged_lb[best_index]:
                best_index = t
                patience = 0
            else:
                patience += 1
            if patience >= max_patience:
                last_iter = t
                break

        alpha = min(e0, e0 * tau / (t + 1))
        lam[t + 1] = lam[t] + alpha * gbar / np.sqrt(vbar + 1e-12)
        last_iter = t

    mu, l = _unpack_cholesky_lambda(lam[best_index], dim)
    l = _regularize_triangular_factor(l)
    covariance = l @ l.T
    covariance = ensure_spd(covariance)
    return GaussianApproximation(
        mean=mu,
        covariance=covariance,
        metadata={"iterations": int(last_iter + 1), "best_index": int(best_index)},
    )


def fit_fgvi_to_gaussian(
    mean: np.ndarray,
    covariance: np.ndarray,
    rank: int,
    seed: int = 123,
    max_iter: int = 1000,
    n_samples: int = 25,
    learning_rate: float = 0.03,
    beta1: float = 0.9,
    beta2: float = 0.999,
    window: int = 50,
    max_patience: int = 50,
) -> GaussianApproximation:
    mean = np.asarray(mean, dtype=float)
    covariance = ensure_spd(covariance)
    rng = np.random.default_rng(seed)
    dim = covariance.shape[0]
    rank = min(rank, dim)
    precision = np.linalg.inv(covariance)
    logdet_target = np.linalg.slogdet(covariance)[1]

    mu = np.zeros(dim, dtype=float)
    b = rng.normal(scale=0.01, size=(dim, rank))
    log_diag = np.zeros(dim, dtype=float)
    lower_bounds = np.full(max_iter, -np.inf, dtype=float)
    averaged_lb = np.full(max_iter, -np.inf, dtype=float)

    params = (mu, b, log_diag)
    first_moments = [np.zeros_like(param) for param in params]
    second_moments = [np.zeros_like(param) for param in params]
    best = tuple(param.copy() for param in params)
    best_index = 0
    patience = 0
    last_iter = 0

    def covariance_from_factor(factor: np.ndarray, log_std: np.ndarray) -> np.ndarray:
        diag_var = np.exp(2.0 * np.clip(log_std, -20.0, 20.0))
        return ensure_spd(factor @ factor.T + np.diag(diag_var))

    def inverse_and_entropy_terms(factor: np.ndarray, log_std: np.ndarray) -> tuple[np.ndarray, float]:
        diag_var = np.exp(2.0 * np.clip(log_std, -20.0, 20.0))
        inv_diag = 1.0 / diag_var
        middle = np.eye(rank) + factor.T @ (inv_diag[:, None] * factor)
        middle_inv = np.linalg.inv(middle)
        inv_cov = np.diag(inv_diag) - (inv_diag[:, None] * factor) @ middle_inv @ (
            factor.T * inv_diag[None, :]
        )
        logdet = float(np.sum(np.log(diag_var)) + np.linalg.slogdet(middle)[1])
        return inv_cov, logdet

    for t in range(1, max_iter + 1):
        diag_std = np.exp(np.clip(log_diag, -20.0, 20.0))
        inv_q, logdet_q = inverse_and_entropy_terms(b, log_diag)
        z = rng.normal(size=(n_samples, rank))
        eps = rng.normal(size=(n_samples, dim))
        theta = mu + z @ b.T + eps * diag_std
        centered = theta - mean
        grad_logp = -(centered @ precision.T)

        grad_mu = grad_logp.mean(axis=0)
        grad_b = grad_logp.T @ z / n_samples + inv_q @ b
        grad_diag_std = (grad_logp * eps).mean(axis=0) + np.diag(inv_q) * diag_std
        grad_log_diag = grad_diag_std * diag_std

        quad = np.einsum("ij,jk,ik->i", centered, precision, centered)
        lower_bounds[t - 1] = float(
            np.mean(-0.5 * (dim * np.log(2.0 * np.pi) + logdet_target + quad))
            + 0.5 * (dim * (1.0 + np.log(2.0 * np.pi)) + logdet_q)
        )

        grads = (grad_mu, grad_b, grad_log_diag)
        for idx, grad in enumerate(grads):
            first_moments[idx] = beta1 * first_moments[idx] + (1.0 - beta1) * grad
            second_moments[idx] = beta2 * second_moments[idx] + (1.0 - beta2) * grad**2
            first_hat = first_moments[idx] / (1.0 - beta1**t)
            second_hat = second_moments[idx] / (1.0 - beta2**t)
            params[idx][...] = params[idx] + learning_rate * first_hat / (np.sqrt(second_hat) + 1e-8)

        log_diag[...] = np.clip(log_diag, -10.0, 10.0)

        if t >= window:
            averaged_lb[t - 1] = lower_bounds[t - window : t].mean()
            if averaged_lb[t - 1] >= averaged_lb[best_index]:
                best_index = t - 1
                best = tuple(param.copy() for param in params)
                patience = 0
            else:
                patience += 1
            if patience >= max_patience:
                last_iter = t
                break
        else:
            best = tuple(param.copy() for param in params)
            best_index = t - 1
        last_iter = t

    mu, b, log_diag = best
    q_cov = covariance_from_factor(b, log_diag)

    return GaussianApproximation(
        mean=mu,
        covariance=q_cov,
        metadata={
            "success": True,
            "message": "stochastic factor Gaussian variational inference",
            "rank": rank,
            "iterations": int(last_iter),
            "best_index": int(best_index),
            "seed": seed,
        },
    )


def sparse_precision_gvi_polypharm(
    log_density: Callable[[np.ndarray], float],
    grad_log_density: Callable[[np.ndarray], np.ndarray],
    dim: int,
    random_effect_dim: int,
    rng: np.random.Generator,
    rho: float = 0.95,
    eps: float = 1e-6,
    max_iter: int = 20000,
    lb_window: int = 10000,
    max_patience: int = 50,
) -> GaussianApproximation:
    block_dim = dim - random_effect_dim
    mu = np.zeros(dim, dtype=float)
    log_a = np.zeros(random_effect_dim, dtype=float)
    c = np.zeros((block_dim, random_effect_dim), dtype=float)
    d_prime = np.zeros((block_dim, block_dim), dtype=float)

    eg2_mu = np.zeros(dim, dtype=float)
    edelta2_mu = np.zeros(dim, dtype=float)
    eg2_log_a = np.zeros(random_effect_dim, dtype=float)
    edelta2_log_a = np.zeros(random_effect_dim, dtype=float)
    eg2_c = np.zeros_like(c)
    edelta2_c = np.zeros_like(c)
    eg2_d = np.zeros_like(d_prime)
    edelta2_d = np.zeros_like(d_prime)

    lower_bounds = np.zeros(max_iter, dtype=float)
    tm = 0
    best_lb = -np.inf
    final_iter = max_iter
    lower_d = np.tril_indices(block_dim)

    def d_from_prime(value: np.ndarray) -> np.ndarray:
        d = np.tril(value, k=-1).copy()
        np.fill_diagonal(d, np.exp(np.diag(value)))
        return d

    for t in range(max_iter):
        s = rng.normal(size=dim)
        s1 = s[:random_effect_dim]
        s2 = s[random_effect_dim:]
        vec_a = np.exp(log_a)
        inv_a = 1.0 / vec_a
        d = d_from_prime(d_prime)
        inv_d_c = solve_triangular(d, c, lower=True, check_finite=False)
        inv_d_c_inv_a = inv_d_c * inv_a[None, :]
        d_inv_t_s2 = solve_triangular(d.T, s2, lower=False, check_finite=False)

        t_inv_t_s = np.concatenate(
            [
                inv_a * s1 - inv_d_c_inv_a.T @ s2,
                d_inv_t_s2,
            ]
        )
        theta = mu + t_inv_t_s

        lower_bounds[t] = (
            log_density(theta)
            + dim / 2.0 * np.log(2.0 * np.pi)
            - np.sum(log_a)
            - np.sum(np.diag(d_prime))
            + 0.5 * float(s @ s)
        )

        if t >= lb_window - 1:
            current = float(lower_bounds[t - lb_window + 1 : t + 1].mean())
            if current < best_lb:
                tm += 1
            else:
                tm = 0
                best_lb = current
            if tm > max_patience:
                final_iter = t + 1
                break

        ts = np.concatenate([vec_a * s1, c @ s1 + d @ s2])
        grad_mu = grad_log_density(theta) + ts
        eg2_mu = rho * eg2_mu + (1.0 - rho) * grad_mu**2
        delta_mu = np.sqrt(edelta2_mu + eps) / np.sqrt(eg2_mu + eps) * grad_mu
        edelta2_mu = rho * edelta2_mu + (1.0 - rho) * delta_mu**2
        mu = mu + delta_mu

        g1 = grad_mu[:random_effect_dim]
        g2 = grad_mu[random_effect_dim:]
        inv_t_grad1 = inv_a * g1
        inv_t_grad2 = solve_triangular(d, g2 - c @ inv_t_grad1, lower=True, check_finite=False)
        x1 = t_inv_t_s[:random_effect_dim]
        x2 = t_inv_t_s[random_effect_dim:]

        grad_log_a = -x1 * inv_t_grad1 * vec_a
        grad_c = -np.outer(x2, inv_t_grad1)
        grad_d = np.tril(-np.outer(x2, inv_t_grad2))
        diag_d = np.diag_indices(block_dim)
        grad_d[diag_d] *= np.diag(d)

        eg2_log_a = rho * eg2_log_a + (1.0 - rho) * grad_log_a**2
        delta_log_a = np.sqrt(edelta2_log_a + eps) / np.sqrt(eg2_log_a + eps) * grad_log_a
        edelta2_log_a = rho * edelta2_log_a + (1.0 - rho) * delta_log_a**2
        log_a = log_a + delta_log_a

        eg2_c = rho * eg2_c + (1.0 - rho) * grad_c**2
        delta_c = np.sqrt(edelta2_c + eps) / np.sqrt(eg2_c + eps) * grad_c
        edelta2_c = rho * edelta2_c + (1.0 - rho) * delta_c**2
        c = c + delta_c

        eg2_d[lower_d] = rho * eg2_d[lower_d] + (1.0 - rho) * grad_d[lower_d] ** 2
        delta_d = np.zeros_like(d_prime)
        delta_d[lower_d] = (
            np.sqrt(edelta2_d[lower_d] + eps) / np.sqrt(eg2_d[lower_d] + eps) * grad_d[lower_d]
        )
        edelta2_d[lower_d] = rho * edelta2_d[lower_d] + (1.0 - rho) * delta_d[lower_d] ** 2
        d_prime = d_prime + delta_d
        d_prime = np.tril(d_prime)

    vec_a = np.exp(log_a)
    inv_a = 1.0 / vec_a
    inv_a_sq = np.diag(inv_a**2)
    d = d_from_prime(d_prime)
    inv_d = solve_triangular(d, np.eye(block_dim), lower=True, check_finite=False)
    inv_d_c_inv_a = (inv_d @ c) * inv_a[None, :]

    covariance = np.zeros((dim, dim), dtype=float)
    covariance[:random_effect_dim, :random_effect_dim] = inv_a_sq + inv_d_c_inv_a.T @ inv_d_c_inv_a
    covariance[random_effect_dim:, :random_effect_dim] = -inv_d.T @ inv_d_c_inv_a
    covariance[:random_effect_dim, random_effect_dim:] = covariance[random_effect_dim:, :random_effect_dim].T
    covariance[random_effect_dim:, random_effect_dim:] = inv_d.T @ inv_d
    covariance = ensure_spd(covariance)

    return GaussianApproximation(
        mean=mu,
        covariance=covariance,
        metadata={"iterations": int(final_iter), "implementation": "block_sparse_precision"},
    )
