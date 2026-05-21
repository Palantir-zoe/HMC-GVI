from __future__ import annotations

import numpy as np


def _autocorrelation_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 2:
        return np.array([1.0])
    variance = np.dot(x, x) / n
    if variance == 0.0:
        return np.ones(1)
    fft_size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, n=fft_size)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:n]
    acov /= np.arange(n, 0, -1)
    return acov / acov[0]


def lag1_autocorrelation(samples: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        acf = _autocorrelation_fft(samples)
        return float(acf[1]) if acf.size > 1 else 0.0
    values = []
    for idx in range(samples.shape[1]):
        acf = _autocorrelation_fft(samples[:, idx])
        values.append(acf[1] if acf.size > 1 else 0.0)
    return float(np.mean(values))


def ess_1d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3:
        return float(n)
    rho = _autocorrelation_fft(x)
    tau = 1.0
    for k in range(1, rho.size - 1, 2):
        pair_sum = rho[k] + rho[k + 1]
        if pair_sum <= 0:
            break
        tau += 2.0 * pair_sum
    return float(n / tau)


def mean_ess(samples: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        return ess_1d(samples)
    return float(np.mean([ess_1d(samples[:, i]) for i in range(samples.shape[1])]))


def efficiency(samples: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=float)
    return mean_ess(samples) / samples.shape[0]


def gaussian_moment_vector(mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    std = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    corr = covariance / np.outer(std, std)
    tril_i, tril_j = np.tril_indices_from(corr, k=-1)
    return np.concatenate([mean, std, corr[tril_i, tril_j]])


def rmse(true_values: np.ndarray, estimate: np.ndarray) -> float:
    true_values = np.asarray(true_values, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    return float(np.sqrt(np.mean((true_values - estimate) ** 2)))

