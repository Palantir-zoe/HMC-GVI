# Result tables

This directory stores small CSV outputs used to document the paper-scale runs.
The samplers do not save raw chains by default.

- `all_repeats.csv` and `all_repeats_summary.csv` are the repeated comparison
  tables produced by `scripts/run_all_repeats.py`.
- `gaussian100_matched_hmc_20260627.csv` and
  `gaussian100_matched_hmc_20260627_summary.csv` record the reported
  HMC/HMC-GVI comparison for the 100-dimensional Gaussian target.

Intermediate tuning grids, smoke-test files, server logs, and one-off debug
outputs are ignored by Git.
