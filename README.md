# AdHOC — Adaptive Hybrid Optimal Control

Code for the **AdHOC** (Adaptive Hybrid Optimal Control) pipeline from the Master's thesis *Adaptive Hybrid Learning for Optimal Control* (Humboldt-Universität zu Berlin, 2026).

The framework addresses the curse of dimensionality in high-dimensional Hamilton–Jacobi–Bellman (HJB) equations by combining supervised data from Pontryagin's Maximum Principle (PMP) with unsupervised physics-informed learning.

## Core methodology

The method is built on three pillars (see thesis Chapter 3):

1. **LQR-residual architecture:** The model learns a nonlinear residual relative to a locally optimal Linear–Quadratic Regulator (LQR) baseline, ensuring local stability at the equilibrium.

2. **Unsupervised training with optional pretraining:** Physics-informed HJB minimisation, optionally preceded by supervised pretraining on PMP-generated costate data.

3. **Adaptive sampling:** Residual-norm–weighted selection of rollout initial conditions (candidate pool size \(N_{\mathrm{cand}}\), batch \(|\mathcal{B}|\)).

## Structure

- **`problems/`** — Collocated PDE models: Burgers, Allen–Cahn, Kuramoto–Sivashinsky.
- **`controls/`** — LQR baseline, residual networks, `train.py`, `model_factory.py`.
- **`data.py`** — PMP / BVP data generation where applicable.
- **`simulation.py`** — Closed-loop integration and Monte Carlo evaluation (optional shared `X0_pool` across runs).
- **`figures.py`** — LaTeX tables (config, training hyperparameters including \(\Delta t_{\min},\Delta t_{\max}\), Monte Carlo with \(\pm\) SE) and plots; default output root is `results/{system}/seed_{seed}/`.
- **`experiments.ipynb`** — End-to-end experiments at repo root; **`experiments.py`** re-exports `create_config` and `figures` for notebook imports.

## Installation

```bash
# Editable install is recommended
pip install -e .
```

## Usage

Training is primarily unsupervised (HJB residual on rollouts), with optional supervised pretraining or a supervised *penalty* on PMP data (`TrainConfig`, `controller_configs` in the notebook). Run `experiments.ipynb` from the **repository root** so paths resolve. Copy or symlink generated assets into your thesis tree if it expects `\input{figures/results/...}` (Chapter~4).

```python
from experiments import create_config

config = create_config(system="burgers", n_states=32, n_controls=2, seed=42)
```

## Citation

If you use this work, please cite the original thesis:

> Söllner, F. (2026). *Adaptive Hybrid Learning for Optimal Control*. Master's Thesis, Institute for Mathematics, Humboldt-Universität zu Berlin.

## License

MIT (see `LICENSE`).
