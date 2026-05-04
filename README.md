# Estimate-Level Adjustment

Code for the paper: **Estimate Level Adjustment For Inference With Proxies Under Random Distribution Shifts**

<!-- TODO: Link to ArXiv paper when available -->

This library provides methods for adjusting proxy-based confidence intervals to account for unobserved heterogeneity across domains. It implements:

- A latent normal model (EM and Method of Moments) for estimating cross-domain heterogeneity
- Leave-one-out validation for computing adjusted confidence intervals
- Coverage calibration curves for evaluating interval quality
- Prediction-Powered Inference (PPI) estimators with covariate shift correction
- Simulation environment for generating synthetic data with controllable covariate shift and concept drift

## Installation

```
pip install -e .
```

For the Civic Comments experiment (requires PyTorch and Transformers):

```
pip install -e ".[ml]"
```

## Quick Start

### Simulation Example

```python
from estimate_level_adjustment import (
    CovShiftDataGenerator,
    CovShiftPPIEstimator,
    AdjustedCovShiftPPIEstimator,
)

# Generate synthetic data with covariate shift
generator = CovShiftDataGenerator(
    n_samples_per_domain=1000,
    n_domains=20,
    concept_drift_degree=0.0,  # Pure covariate shift
    random_seed=42,
)
df = generator.generate_data()

# Compute PPI estimates
estimator = CovShiftPPIEstimator(
    df=df[df["domain"] == 20],  # Target domain
    primary_outcome_column="primary_outcome",
    proxy_outcome_column="proxy_outcome",
    importance_weight_column="importance_weight",
)
ppi_estimate = estimator.compute_ppi_mean_estimate(weighted=True)
print(ppi_estimate)
```

### Adjustment Example

```python
from estimate_level_adjustment import fit_latent_normal
import numpy as np

# Observed domain-level differences and their standard errors
Y = np.array([0.02, -0.01, 0.05, 0.03, -0.02])  # primary - proxy differences
sigma = np.array([0.01, 0.02, 0.015, 0.01, 0.025])  # standard errors

# Fit the latent normal model
result = fit_latent_normal(Y, sigma, method="MoM")
print(f"Mean bias (rho): {result.rho:.4f}")
print(f"Heterogeneity (gamma^2): {result.gamma_squared:.6f}")
```

## Notebooks

The `notebooks/` directory contains Jupyter notebooks reproducing the paper results:

1. **`simulation_experiments.ipynb`** — Simulation studies with controllable covariate shift and concept drift
2. **`train_toxicity_model.ipynb`** — Training a BERT-based toxicity classifier on the Civil Comments dataset
3. **`civic_comments_results.ipynb`** — Applying the adjustment methodology to the Civil Comments public dataset

## Structure

```
estimate_level_adjustment/
    adjustment.py              # Latent normal model (MoM/MLE)
    adjustment_model.py        # EM algorithm for latent normal
    adjusted_ppi_estimator.py  # PPI estimator with estimate-level adjustment
    cov_shift_estimator.py     # Base PPI estimator with covariate shift
    data_generators.py         # Simulation data generator
    civil_comments_model.py    # BERT toxicity classifier
    summarize_results.py       # Result aggregation and plotting
    kaggle_dataset.py          # Kaggle dataset loader
    utils.py                   # Leave-one-out validation and coverage curves
    stats_utils.py             # Statistical utilities (CI, Estimate dataclass)
```

## Citation

If you use this code, please cite our paper:

<!-- TODO: Update with ArXiv link when available -->

```bibtex
@article{estimate_level_adjustment,
  title={Estimate Level Adjustment For Inference With Proxies Under Random Distribution Shifts},
  author={Meta Platforms, Inc.},
  year={2026}
}
```

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.
