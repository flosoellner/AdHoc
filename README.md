# ADHOC: Adaptive Hybrid Learning for Optimal Control

This repository contains the implementation of the **ADHOC** (Adaptive Hybrid Learning for Optimal Control) framework, as developed in the Master's Thesis: *"Adaptive Hybrid Learning for Optimal Control"* (Humboldt-Universität zu Berlin, 2026).

The framework addresses the curse of dimensionality in high-dimensional Hamilton–Jacobi–Bellman (HJB) equations by combining supervised data from Pontryagin's Maximum Principle (PMP) with unsupervised physics-informed learning.

## Core methodology

The AdHOC framework is built on three pillars:

1. **LQR-residual architecture:** The model learns a nonlinear residual relative to a locally optimal Linear–Quadratic Regulator (LQR) baseline, ensuring local stability at the equilibrium.

2. **Unsupervised training with optional pretraining:** Physics-informed HJB minimisation, optionally preceded by supervised pretraining on PMP-generated costate data.

3. **Adaptive sampling:** Steepness-aware importance sampling that concentrates training on dynamically challenging regions of the state space.

## Structure

- **`problems/`** — Nonlinear PDE benchmarks: Burgers equation, Allen–Cahn equation, Korteweg–de Vries (KdV).
- **`controls/`** — LQR-residual gradient learning: `lqr.py`, `nn.py`, `train.py`, `model_factory.py`.
- **`data.py`** — PMP-based data generation (BVP solves) for supervised pretraining.
- **`simulation.py`** — Closed-loop rollout and Monte Carlo evaluation.
- **`figures.py`** — Thesis-ready LaTeX tables and plots.
- **`experiments/`** — Jupyter notebooks for Burgers and Allen–Cahn experiments.

## Installation

```bash
# Editable install is recommended to use the 'experiments' scripts
pip install -e .
```

## Usage

Training is unsupervised (HJB residual along rollouts), with an optional supervised pretraining phase on PMP-generated state–costate pairs (`pretrain=True`). Run the notebooks in `experiments/` (e.g. `burgers.ipynb`, `allen_cahn.ipynb`) from the project root or from the `experiments` directory.

```python
from experiments import create_config

config = create_config(system="burgers", n_states=32, n_controls=2, seed=42)
```

## Citation

If you use this work, please cite the original thesis:

> Söllner, F. (2026). *Adaptive Hybrid Learning for Optimal Control*. Master's Thesis, Institute for Mathematics, Humboldt-Universität zu Berlin.

## License

MIT (see `LICENSE`).
