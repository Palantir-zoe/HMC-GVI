# Data Sources

This directory stores small public datasets used by the numerical experiments.

## `polypharmacy.csv`

The polypharmacy data correspond to the `polypharm` dataset in the CRAN R
package `aplore3`, an unofficial companion package for Hosmer, Lemeshow and
Sturdivant, *Applied Logistic Regression*, 3rd ed. (Wiley, 2013). The original
data description in `aplore3` lists 3,500 observations from 500 subjects
measured over seven years, with a binary response indicating whether a subject
took drugs from more than three different classes.

The GLMM experiment in this repository follows the polypharmacy logistic
random-intercept example used by Tan and Nott (2018), "Gaussian variational
approximation with sparse precision matrices". The CSV file here is an exported
copy of the `aplore3::polypharm` data so that the Python scripts can be run
without requiring an R installation.

Upstream references:

- `aplore3::polypharm` documentation:
  <https://rdrr.io/cran/aplore3/man/polypharm.html>
- CRAN package `aplore3`: <https://cran.r-project.org/package=aplore3>
- Tan and Nott (2018) arXiv source:
  <https://arxiv.org/abs/1605.05622>

The `aplore3` package is distributed under GPL-3. This repository uses the CSV
only as a reproducibility copy for the experiments; please consult the upstream
package and cited book/paper for authoritative data documentation.
