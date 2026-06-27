# HMC-GVI

Python implementation for the numerical experiments on using Gaussian
variational inference (GVI) to construct covariance preconditioners for HMC and
MALA.

The reusable code lives in `src/`; paper-scale experiment entry points live in
`scripts/`. The scripts write compact CSV summaries and do not store raw Markov
chains by default.

## Repository Layout

- `src/mcmc.py`: RMH, adaptive MH, MALA, and HMC samplers.
- `src/vi.py`: Gaussian variational approximations used for covariance
  estimation.
- `src/targets.py`: posterior log densities and gradients.
- `src/tuning.py`: MALA and HMC tuning utilities.
- `src/metrics.py`: ESS, normalized efficiency, and autocorrelation metrics.
- `src/data.py`: dataset loaders.
- `scripts/run_all_repeats.py`: repeated comparison tables for the logistic
  regression, Gaussian, and GLMM examples.
- `scripts/run_gaussian100_matched.py`: reported 100-dimensional Gaussian
  HMC/HMC-GVI comparison.
- `scripts/run_glmm_hmc_comparison.py`: reported GLMM HMC/HMC-GVI comparison.
- `data/`: CSV datasets used by the experiments.
- `results/`: curated result tables.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
```

The code requires Python 3.10 or newer. The main dependencies are NumPy,
Pandas, and SciPy.

If you use `uv`, the locked environment can be installed with:

```bash
uv sync
```

## Reproducing Results

The paper-scale settings use `n_samples=1,000,000`, `burn_in=10,000`, and
`base_seed=123`. Runtime depends strongly on hardware, especially for the GLMM
example.

Hyperparameters are selected before the final sampling runs and then kept
unchanged during sampling, so no adaptation is performed on the retained chains.

Run the full repeated comparison table:

```bash
python scripts/run_all_repeats.py \
  --repeats 3 \
  --targets all \
  --output results/all_repeats.csv \
  --summary-output results/all_repeats_summary.csv
```

Run only a subset:

```bash
python scripts/run_all_repeats.py --repeats 3 --targets pima,german
python scripts/run_all_repeats.py --repeats 3 --targets gaussian_100d
python scripts/run_all_repeats.py --repeats 3 --targets glmm
```

Reproduce the reported 100-dimensional Gaussian HMC/HMC-GVI comparison:

```bash
python scripts/run_gaussian100_matched.py \
  --repeats 3 \
  --output results/gaussian100_matched_hmc_20260627.csv \
  --summary-output results/gaussian100_matched_hmc_20260627_summary.csv
```

Reproduce the reported GLMM HMC/HMC-GVI comparison:

```bash
python scripts/run_glmm_hmc_comparison.py \
  --repeats 3 \
  --output results/glmm_hmc_comparison_20260627.csv \
  --summary-output results/glmm_hmc_comparison_20260627_summary.csv
```

For quick smoke tests, override the chain lengths:

```bash
python scripts/run_gaussian100_matched.py \
  --repeats 1 \
  --methods HMC-GVI \
  --n-samples 100 \
  --burn-in 10 \
  --fgvi-n-samples 20 \
  --output results/smoke_gaussian100.csv \
  --summary-output results/smoke_gaussian100_summary.csv
```

## Seeds

The scripts use deterministic seeds derived from `base_seed=123`, the repeat
index, dataset offsets, and method offsets. For example, in the GLMM experiment
the HMC repeats use seeds `14176`, `24176`, and `34176`, while the HMC-GVI
repeats use `14190`, `24190`, and `34190`.

## Result Files

The repository keeps small CSV tables under `results/`. Long logs, tuning grids,
one-off server outputs, and smoke-test outputs are ignored by Git. Runtime
numbers may vary across machines, but the reported efficiency, acceptance, and
autocorrelation metrics should be close when the same seeds and settings are
used.

## License

This code is released under the MIT License. See `LICENSE`.
