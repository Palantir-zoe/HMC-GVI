from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .utils import ensure_spd


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
        l = ensure_spd(l @ l.T)
        chol = np.linalg.cholesky(l)
        inv_chol = np.linalg.solve(chol, np.eye(dim))

        eps = rng.normal(size=(n_samples, dim))
        theta = mu + eps @ chol.T

        grad = np.zeros(glen, dtype=float)
        lb = 0.0
        for i in range(n_samples):
            th = theta[i]
            value = (
                log_density(th)
                + np.log(np.abs(np.linalg.det(chol)))
                + 0.5 * (th - mu) @ inv_chol.T @ inv_chol @ (th - mu)
            )
            lb += value / n_samples
            gs = grad_log_density(th) + inv_chol.T @ inv_chol @ (th - mu)
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
) -> GaussianApproximation:
    mean = np.asarray(mean, dtype=float)
    covariance = ensure_spd(covariance)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    rank = min(rank, covariance.shape[0])

    top_vals = np.clip(eigvals[:rank], 0.0, None)
    top_vecs = eigvecs[:, :rank]
    b = top_vecs @ np.diag(np.sqrt(top_vals))
    low_rank = b @ b.T
    diagonal_residual = np.clip(np.diag(covariance - low_rank), 1e-8, None)
    q_cov = ensure_spd(low_rank + np.diag(diagonal_residual))

    return GaussianApproximation(
        mean=mean.copy(),
        covariance=q_cov,
        metadata={
            "success": True,
            "message": "spectral low-rank plus diagonal approximation",
            "rank": rank,
            "max_iter": max_iter,
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
    mu = np.zeros(dim, dtype=float)
    t_mat = np.eye(dim, dtype=float)
    t_prime = np.eye(dim, dtype=float)
    np.fill_diagonal(t_prime, np.log(np.diag(t_prime)))

    eg2_mu = np.zeros(dim, dtype=float)
    edelta2_mu = np.zeros(dim, dtype=float)
    eg2_t = np.zeros((dim, dim), dtype=float)
    edelta2_t = np.zeros((dim, dim), dtype=float)

    lower_bounds = np.zeros(max_iter, dtype=float)
    tm = 0
    best_lb = -np.inf
    final_iter = max_iter

    for t in range(max_iter):
        s = rng.normal(size=dim)
        s1 = s[:random_effect_dim]
        s2 = s[random_effect_dim:]
        a = t_mat[:random_effect_dim, :random_effect_dim]
        c = t_mat[random_effect_dim:, :random_effect_dim]
        d = t_mat[random_effect_dim:, random_effect_dim:]

        vec_a = np.diag(a)
        inv_a = np.diag(1.0 / vec_a)
        inv_d = np.linalg.inv(d)
        inv_d_c_inv_a = (inv_d @ c) * (1.0 / vec_a)

        t_inv_t_s = np.concatenate(
            [
                (1.0 / vec_a) * s1 - inv_d_c_inv_a.T @ s2,
                inv_d.T @ s2,
            ]
        )
        theta = mu + t_inv_t_s

        lower_bounds[t] = (
            log_density(theta)
            + dim / 2.0 * np.log(2.0 * np.pi)
            - np.sum(np.log(np.abs(vec_a)))
            - np.log(np.abs(np.linalg.det(d)))
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
        inv_t_grad = np.concatenate([(1.0 / vec_a) * g1, -inv_d_c_inv_a @ g1 + inv_d @ g2])
        grad_t_prime = -np.outer(t_inv_t_s, inv_t_grad)
        diag_idx = np.diag_indices(dim)
        grad_t_prime[diag_idx] *= np.diag(t_mat)

        eg2_t = rho * eg2_t + (1.0 - rho) * grad_t_prime**2
        delta_t_prime = np.sqrt(edelta2_t + eps) / np.sqrt(eg2_t + eps) * grad_t_prime
        edelta2_t = rho * edelta2_t + (1.0 - rho) * delta_t_prime**2
        t_prime = t_prime + delta_t_prime

        t_prime[:random_effect_dim, :random_effect_dim] = np.diag(
            np.diag(t_prime[:random_effect_dim, :random_effect_dim])
        )
        upper = np.triu_indices(dim, k=1)
        t_prime[upper] = 0.0

        t_mat = t_prime.copy()
        np.fill_diagonal(t_mat, np.exp(np.diag(t_mat)))

    a = t_mat[:random_effect_dim, :random_effect_dim]
    c = t_mat[random_effect_dim:, :random_effect_dim]
    d = t_mat[random_effect_dim:, random_effect_dim:]
    vec_a = np.diag(a)
    inv_a_sq = np.diag((1.0 / vec_a) ** 2)
    inv_d = np.linalg.inv(d)
    inv_d_c_inv_a = (inv_d @ c) * (1.0 / vec_a)

    covariance = np.zeros((dim, dim), dtype=float)
    covariance[:random_effect_dim, :random_effect_dim] = inv_a_sq + inv_d_c_inv_a.T @ inv_d_c_inv_a
    covariance[random_effect_dim:, :random_effect_dim] = -inv_d.T @ inv_d_c_inv_a
    covariance[:random_effect_dim, random_effect_dim:] = covariance[random_effect_dim:, :random_effect_dim].T
    covariance[random_effect_dim:, random_effect_dim:] = inv_d.T @ inv_d
    covariance = ensure_spd(covariance)

    return GaussianApproximation(
        mean=mu,
        covariance=covariance,
        metadata={"iterations": int(final_iter)},
    )
