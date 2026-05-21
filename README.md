# HMC-GVI

This repository contains a cleaned Python implementation of the HMC-GVI
experiments.  The code is organized as flat reusable modules under `src/` plus
runnable experiment scripts under `scripts/`, so the main algorithms are
separated from the long-running numerical experiments.

## Layout

- `src/`: reusable implementation code.
  - `mcmc.py`: RMH, adaptive MH, MALA, and HMC samplers.
  - `vi.py`: Gaussian variational approximations used for covariance estimation.
  - `targets.py`: posterior log densities and gradients.
  - `tuning.py`: MALA and HMC tuning utilities.
  - `metrics.py`: ESS, efficiency, autocorrelation, and moment metrics.
  - `data.py`: dataset loaders.
- `scripts/`: experiment entry points.
  - `run_all_repeats.py`: final repeated comparison table.
  - `run_pima.py`
  - `run_german.py`
  - `run_gaussian_100d.py`
  - `run_glmm.py`
  - `summarize_results.py`
- `data/`: CSV datasets used by the experiments.
- `results/`: final generated result tables.
  - `all_repeats.csv`: final per-repeat results.
  - `all_repeats_summary.csv`: final averaged summary.

## Setup

From the repository root:

```bash
python -m pip install -e .
```

## Running Experiments

Run a single experiment:

```bash
python scripts/run_pima.py
python scripts/run_german.py
python scripts/run_gaussian_100d.py
python scripts/run_glmm.py
```

Run the repeated comparison table:

```bash
python scripts/run_all_repeats.py --repeats 3 --targets all
```

This is the canonical command used for the final committed tables:

- `base_seed = 123`
- `n_samples = 1,000,000`
- `burn_in = 10,000`
- `pre_burn = 10,000`
- MALA/HMC burn-covariance pre-burn is `10,000` by default.
- Pima HMC-GVI uses the final `pima_hmc_gvi_lcap24` tuning profile.
- The final GLMM MALA baseline uses burn-covariance pre-burn `100,000`.
- Gaussian and GLMM HMC-GVI use the final `hmc_gvi_grid_rho_constrained`
  tuning profile.

Run only a subset:

```bash
python scripts/run_all_repeats.py --repeats 3 --targets pima,german
python scripts/run_all_repeats.py --repeats 3 --targets glmm
```

## Notes

The default settings follow paper-scale chains and can be slow.  The scripts
write CSV outputs under `results/`; only the final `all_repeats*.csv` tables are
kept in the repository.
