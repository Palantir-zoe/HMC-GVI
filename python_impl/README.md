# Python Reimplementation Notes

This directory contains a Python reimplementation of the HMC-GVI numerical experiments from the paper. The experiment scripts use the paper-scale main-chain settings directly; there is no separate runtime mode.

## Experiment Mapping

- `python_impl/experiments/exp_4_1_gaussian_speed_accuracy.py`
  - Corresponds to the Gaussian speed/accuracy comparison for FGVI versus MCMC.
  - Uses dimensions `2, 10, 50, 100, 500`.
- `python_impl/experiments/exp_4_2_logistic_regression.py`
  - Corresponds to posterior sampling for the Pima and German logistic regression examples.
  - Compares RMH, AM, MALA, HMC, and HMC-GVI.
  - The preliminary covariance-estimation runs for RMH, MALA, and HMC use `10000` samples.
- `python_impl/experiments/exp_4_2_gaussian_hmc_gvi.py`
  - Corresponds to the 100-dimensional Gaussian HMC-GVI experiment.
  - Uses FGVI with rank `5` for the HMC-GVI covariance.
  - The preliminary covariance-estimation runs use `10000` samples.
- `python_impl/experiments/exp_4_2_glmm.py`
  - Corresponds to the Polypharmacy logistic GLMM experiment.
  - Uses sparse-precision GVI for MALA-GVI and HMC-GVI.
  - The preliminary covariance-estimation runs for MALA and HMC use `10000` samples.

## Core Modules

- `python_impl/data.py`: loads the Pima, German, and Polypharmacy datasets.
- `python_impl/targets.py`: defines log posterior densities and gradients.
- `python_impl/vi.py`: implements CGVI, Gaussian-target FGVI, and sparse-precision GVI.
- `python_impl/mcmc.py`: implements RMH, AM, MALA, and HMC.
- `python_impl/metrics.py`: computes efficiency, lag-1 autocorrelation, and Gaussian moment RMSE.

## Running Experiments

Run commands from the repository root:

```bash
python -m python_impl.experiments.exp_4_1_gaussian_speed_accuracy
python -m python_impl.experiments.exp_4_2_logistic_regression
python -m python_impl.experiments.exp_4_2_gaussian_hmc_gvi
python -m python_impl.experiments.exp_4_2_glmm
```

The default output files are:

- `python_impl/results_exp_4_1.csv`
- `python_impl/results_exp_4_2_logistic.csv`
- `python_impl/results_exp_4_2_gaussian.csv`
- `python_impl/results_exp_4_2_glmm.csv`

These runs are intentionally expensive because they follow the paper-scale sampling settings.

Result tables include `covariance_samples` and `covariance_source` columns where a method estimates a covariance matrix from preliminary MCMC samples.
