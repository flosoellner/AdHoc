import numpy as np


def sample_conditions(config, n: int, dist: float = None, seed: int = None, K: int = 10):
    """
    Sample n initial conditions via config.ocp.sample_initial_conditions;
    optionally normalize to dist. K is passed through for problems that use it
    (e.g. Fourier modes).
    """
    ocp = config.ocp
    X0 = ocp.sample_initial_conditions(n, seed=seed, K=K)

    if dist is not None:
        norm_func = config.norm
        X0_norm = norm_func(X0).reshape(1, -1)
        X0 = X0 * (float(dist) / (X0_norm + 1e-12))

    return X0  # (d, n)


def adaptive_sample_conditions(
    config,
    n: int,
    *,
    controller,
    n_candidates: int | None = None,
    dist: float | None = None,
    seed: int | None = None,
    device: str = "cpu",
):
    """
    Sample `n_candidates` using `sample_conditions`, score by ||dVdX - dVdX_LQR|| (residual), return top-n.
    n_candidates must be set by the experiment.

    controller can be:
    - LQR: has .eval_dVdX(X) with X shaped (d,N) numpy
    - NN Control with GradNet: pass ctrl (has .grad_net torch module)
    """
    if n_candidates is None:
        raise ValueError("n_candidates must be set by the experiment (e.g. n_candidates=200)")
    Xcand = sample_conditions(config, n_candidates*n, seed=seed)  # (d,N)

    # get gradients (d,N)
    if hasattr(controller, "eval_dVdX"):
        G = controller.eval_dVdX(Xcand)
    elif hasattr(controller, "grad_net"):
        import torch
        x = torch.tensor(Xcand.T, dtype=torch.float32, device=device)  # (N,d)
        with torch.no_grad():
            G = controller.grad_net(x).detach().cpu().numpy().T        # (d,N)
    elif callable(controller):
        import torch
        x = torch.tensor(Xcand.T, dtype=torch.float32, device=device)
        with torch.no_grad():
            G = controller(x).detach().cpu().numpy().T   # (d,N)
    else:
        raise TypeError("controller must have eval_dVdX (LQR) or grad_net (NN Control).")

    # Get LQR gradient and compute residual
    G_lqr = config.ocp.LQR.eval_dVdX(Xcand)  # (d,N)
    residual = G - G_lqr  # (d,N)
    scores = np.linalg.norm(residual, axis=0)
    
    # Filter out non-finite values
    valid_mask = np.isfinite(scores) & np.isfinite(Xcand).all(axis=0)
    if not valid_mask.any():
        # If no valid candidates, return the first n candidates (even if invalid)
        # This will be caught later in data generation
        idx = np.arange(min(n, Xcand.shape[1]))
    else:
        # Only consider valid candidates for scoring
        valid_scores = scores[valid_mask]
        valid_idx = np.where(valid_mask)[0]
        
        # Sort valid candidates by score (highest first)
        sorted_valid_idx = np.argsort(valid_scores)[::-1]
        
        # Take top n valid candidates
        n_valid = min(n, len(sorted_valid_idx))
        idx = valid_idx[sorted_valid_idx[:n_valid]]
        
        # If we don't have enough valid candidates, pad with remaining valid ones
        if len(idx) < n:
            remaining_valid = valid_idx[sorted_valid_idx[n_valid:]]
            if len(remaining_valid) > 0:
                n_needed = n - len(idx)
                idx = np.concatenate([idx, remaining_valid[:n_needed]])
            # If still not enough, we'll just return what we have
    
    return Xcand[:, idx], scores[idx]
