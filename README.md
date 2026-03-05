<div align="center">

# Quantile2SpaceTime
### ML quantiles → latent Gaussian fields → coherent spatio-temporal simulation

A research codebase for **modeling and simulating spatio-temporal processes** by combining  
**machine-learning quantile regression** with **latent Gaussian random fields (GRFs)**.

</div>

---

## Paper

This repository accompanies the HAL preprint:

**Combining machine learning quantile regression and Gaussian random fields: a general framework for modeling and simulating space-time processes**  
https://hal.science/hal-05441043/

---

## Overview

**Quantile2SpaceTime** implements a coherent two-stage framework:

1. **Conditional marginals (data space)**  
   Learn the conditional distribution \(Y \mid X\) using quantile regression:
   - **KNN CDF**
   - **Quantile Regression Forests (QRF)**
   - **Quantile Regression Neural Networks (QRNN)**

2. **Dependence (latent space)**  
   Map observations to a latent Gaussian space and model dependence with a GRF:
   - Transform: \(U = F_{Y|X}(y)\), then \(Z = \Phi^{-1}(U)\)
   - Fit a spatio-temporal GRF for \(Z(s,t)\) (e.g., **Matérn–Gneiting**)
   - Simulate \(Z\) and invert back to \(Y\)

This yields simulations that preserve:
- **non-Gaussian, covariate-dependent marginals**, and
- **spatio-temporal dependence** through the latent GRF.

---

## What’s inside

### Marginal model: `SitewiseMarginal`
- Methods: `knn`, `qrf`, `qrnn`
- Optional variable selection
- Time-series cross-validation (pinball loss)
- Public API:
  - `predict_quantiles(X)`
  - `predict_cdf(X, Y_eval)`
  - `y_to_z(X, Y)`
  - `z_to_y(X, Z)`

### Latent GRF model: `GneitingModel`
- Composite-likelihood estimation for Matérn–Gneiting space-time covariance
- Block strategies: `random`, `anchor`, `balanced`
- Option to fix or estimate Matérn smoothness `nu`

### Latent simulation: `simulate_gneiting_jax`
- JAX-based simulation with chunking (scales to large numbers of spectral draws)

### Orchestrator (pipeline)
- End-to-end fitting and simulation:
  - fit marginals
  - map \(Y \leftrightarrow Z\)
  - fit GRF parameters
  - simulate \(Z\), invert to \(Y\)

---

## Quickstart

> Replace `quantile2spacetime` / `Quantile2SpaceTimeModel` with your actual module/class names.

```python
import numpy as np
from quantile2spacetime.pipeline import Quantile2SpaceTimeModel

# coords: (n_sites, d)
# X_cov : (n_time, n_features)
# Y_obs : (n_time, n_sites)

model = Quantile2SpaceTimeModel(
    coords=coords,
    # Marginal stage
    marginal_method="qrf",                 # "knn" | "qrf" | "qrnn"
    marginal_kwargs={"n_jobs_sites": 4},
    var_select=False,
    # Latent GRF stage
    gneiting_strategy="balanced",
    gneiting_estimate_nu=False,
)

model.fit(X_cov=X_cov, Y_obs=Y_obs, dates=dates)

wt_seq, Zsim, Ysim = model.simulate(
    X_test=X_test,
    test_dates=test_dates,
    X_train=X_cov,
    Y_train=Y_obs,
    train_dates=dates,
    n_simulations=20,
    L_draws=50_000,
    chunk_size=500,
)
