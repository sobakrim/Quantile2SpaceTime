# MLQuantile4SpaceTime
Machine-learning quantile regression for space-time processes

[![HAL](https://img.shields.io/badge/HAL-hal--05441043-B03532)](https://hal.science/hal-05441043/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research codebase for **modeling and simulating spatio-temporal processes** by combining
**machine-learning quantile regression** with **latent Gaussian random fields (GRFs)**.

---

## Paper

- HAL preprint: https://hal.science/hal-05441043/

---

## Features

- **Conditional marginals** \(Y \mid X\) via quantile regression:
  - KNN-based conditional CDF (`knn`)
  - Quantile Regression Forests (`qrf`)
  - Quantile Regression Neural Networks (`qrnn`, via `quantnn`)
- **Latent Gaussian mapping**: \(U = F_{Y|X}(y)\), \(Z = \Phi^{-1}(U)\)
- **Spatio-temporal dependence** in latent space with GRFs (e.g., Matérn–Gneiting)
- End-to-end **fit → simulate → invert** workflow
- Optional hyperparameter selection via time-series cross-validation (depending on method)

---

## Installation

### From GitHub (requires `git`)
```bash
pip install "git+https://github.com/sobakrim/MLQuantile4SpaceTime.git@main"
