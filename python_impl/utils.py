from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]


def now() -> float:
    return time.perf_counter()


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return expit(x)


def log1pexp(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def standardize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0)
    stds[stds == 0.0] = 1.0
    return (x - means) / stds


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(x.shape[0]), x])


def ensure_spd(matrix: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    eye = np.eye(matrix.shape[0])
    scale = jitter
    for _ in range(12):
        try:
            np.linalg.cholesky(matrix + scale * eye)
            return matrix + scale * eye
        except np.linalg.LinAlgError:
            scale *= 10.0
    raise np.linalg.LinAlgError("Matrix could not be regularized to SPD.")


def draw_gaussian_covariance(
    dim: int,
    rng: np.random.Generator,
    off_diag: float = 0.8,
) -> np.ndarray:
    while True:
        cov = np.full((dim, dim), off_diag, dtype=float)
        np.fill_diagonal(cov, rng.gamma(shape=2.0, scale=3.0, size=dim))
        try:
            np.linalg.cholesky(cov)
            return cov
        except np.linalg.LinAlgError:
            continue


def time_call(func: Callable, *args, **kwargs):
    start = now()
    result = func(*args, **kwargs)
    elapsed = now() - start
    return result, elapsed

