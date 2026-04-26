from __future__ import annotations

import numpy as np

from .utils import log1pexp, sigmoid


def gaussian_log_density(theta: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    delta = theta - mean
    precision = np.linalg.solve(covariance, np.eye(covariance.shape[0]))
    return float(-0.5 * delta @ precision @ delta)


def gaussian_grad_log_density(theta: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    return -np.linalg.solve(covariance, theta - mean)


def logistic_loglik(beta: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    eta = x @ beta
    return float(np.sum(y * eta - log1pexp(eta)))


def logistic_logposterior(beta: np.ndarray, y: np.ndarray, x: np.ndarray, prior_sd: float = 10.0) -> float:
    prior_var = prior_sd**2
    return logistic_loglik(beta, y, x) - 0.5 * np.sum(beta**2) / prior_var


def logistic_grad_logposterior(
    beta: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    prior_sd: float = 10.0,
) -> np.ndarray:
    probs = sigmoid(x @ beta)
    prior_var = prior_sd**2
    return x.T @ (y - probs) - beta / prior_var


def polypharm_logposterior(theta: np.ndarray, data: dict[str, np.ndarray]) -> float:
    n_subjects = data["n_subjects"]
    repeats = data["repeats"]
    x = data["x"]
    y = data["y"]
    sb = 10.0
    sz = 10.0

    u = theta[:n_subjects]
    beta = theta[n_subjects : n_subjects + 8]
    zeta = theta[-1]
    u_expanded = np.repeat(u, repeats)
    eta = x @ beta + u_expanded

    loglik = np.sum(y * eta - log1pexp(eta))
    random_effect_prior = -0.5 * np.exp(-2.0 * zeta) * np.sum(u**2) - n_subjects * zeta
    beta_prior = -0.5 * np.sum(beta**2) / (sb**2)
    zeta_prior = -0.5 * zeta**2 / (sz**2)
    return float(loglik + random_effect_prior + beta_prior + zeta_prior)


def polypharm_grad_logposterior(theta: np.ndarray, data: dict[str, np.ndarray]) -> np.ndarray:
    n_subjects = data["n_subjects"]
    repeats = data["repeats"]
    x = data["x"]
    y = data["y"]

    u = theta[:n_subjects]
    beta = theta[n_subjects : n_subjects + 8]
    zeta = theta[-1]
    u_expanded = np.repeat(u, repeats)
    eta = x @ beta + u_expanded
    probs = sigmoid(eta)

    residual = y - probs
    grad_u = residual.reshape(n_subjects, repeats).sum(axis=1) - np.exp(-2.0 * zeta) * u
    grad_beta = x.T @ residual - beta / 100.0
    grad_zeta = np.exp(-2.0 * zeta) * np.sum(u**2) - zeta / 100.0 - n_subjects
    return np.concatenate([grad_u, grad_beta, np.array([grad_zeta])])

