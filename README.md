# ADHOC: Adaptive Hybrid Learning for Optimal Control

This repository contains the implementation of the **ADHOC** (Adaptive Hybrid Learning for Optimal Control) framework, as developed in the Master's Thesis: *"Adaptive Hybrid Learning for Optimal Control"* (Humboldt-Universität zu Berlin, 2026).

The framework addresses the **curse of dimensionality** in high-dimensional Hamilton-Jacobi-Bellman (HJB) equations by combining supervised expert data from Pontryagin's Maximum Principle (PMP) with unsupervised physics-informed learning.

## Core Methodology

The ADHOC framework is built on three pillars:

1. **LQR-Residual Architecture:** Instead of learning the full value function, the model learns a nonlinear residual relative to a locally optimal **Linear-Quadratic Regulator (LQR)** baseline. This ensures local stability at the equilibrium and reduces approximation complexity.

2. **Hybrid Training:** A synergy of **supervised pretraining** (imitating PMP costates) and **unsupervised HJB refinement** (minimizing the Bellman residual).

3. **Adaptive Sampling:** A physics-informed heuristic that concentrates training samples in "steep" or dynamically challenging regions of the state space, creating an automated curriculum.

## Structure

- **`problems/`** — Nonlinear PDE benchmarks:
  - **Burgers' Equation:** High-dimensional control with viscous diffusion.
  - **Allen–Cahn Equation:** Bistable reaction-diffusion dynamics.
  - **Korteweg-De Vries (KdV):** Dispersive wave control.

- **`controls/`** — Implementation of the **LQR-residual gradient** learning framework:
  - `nn.py`: Gradient-based neural parameterization.
  - `train.py`: Hybrid loss implementation (Supervised + Physics-Informed).

- **`data.py`** — Generation of **PMP-based expert trajectories** for supervised anchoring.

- **`simulation.py`** — Closed-loop rollout simulation and Monte Carlo evaluation.

- **`figures.py`** — Generation of thesis-ready LaTeX tables and plots.

- **`experiments/`** — Rigorous testing environments and Jupyter notebooks for benchmarks.

## Installation

```bash
# Editable install is recommended to use the 'experiments' scripts
pip install -e .
```

## Usage

### Training a Hybrid Controller

The framework typically follows a two-phase training curriculum:

1. **Phase I (Supervised):** Warm-start the network using costates from PMP-generated data.

2. **Phase II (Unsupervised):** Refine the controller using HJB residuals along simulated rollouts with **steepness-aware importance sampling**.

```python
from experiments import create_config

# Configure a Burgers' equation experiment
config = create_config(
    system="burgers", 
    n_states=32, 
    adaptive_sampling=True, 
    hybrid_lambda=0.1
)
```

## Citation

If you use this work, please cite the original thesis:

> Söllner, F. (2026). *Adaptive Hybrid Learning for Optimal Control*. Master's Thesis, Institute for Mathematics, Humboldt-Universität zu Berlin.

## License

MIT (see `LICENSE`).
