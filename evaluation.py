import numpy as np
import warnings
import json, hashlib, os
from simulation import sim_closed_loop
from sampling import sample_conditions
import torch
import os

# --- Core Simulation Wrappers ---

def rollout_success(config, controller, x0, eps=1e-2):
    """
    Single source of truth for checking if a rollout succeeded.
    """
    # Suppress warnings for cleaner logs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        old_settings = np.seterr(over="ignore", invalid="ignore", divide="ignore")
        
        try:
            t, X, status = sim_closed_loop(
                config.ocp.dynamics,
                config.ocp.closed_loop_jacobian,
                controller,
                tspan=[0.0, config.t1_initial],
                X0=np.asarray(x0, dtype=float),
                solver='LSODA',
                atol=1e-06,
                rtol=1e-03,
                events=config.ocp.make_integration_events(),
            )
        finally:
            np.seterr(**old_settings)

    if status < 0 or np.isnan(X).any(): 
        return False, np.inf
    
    norms = config.norm(X)
    # Check if we hit the target region
    is_stable = np.any(norms <= eps)
    
    t_conv = np.inf
    if is_stable:
        # Find first time we entered epsilon ball
        idx = np.where(norms <= eps)[0]
        t_conv = float(t[idx[0]])
        
    return is_stable, t_conv

def stability_score(config, controller, X0_pool, eps=1e-2):
    n = X0_pool.shape[1]
    if n == 0: return {"S": 0.0, "tconv_med": np.nan}
    
    successes = 0
    tconvs = []
    
    for i in range(n):
        ok, tconv = rollout_success(config, controller, X0_pool[:, i], eps=eps)
        if ok:
            successes += 1
            tconvs.append(tconv)
            
    return {
        "S": successes / n,
        "tconv_med": float(np.median(tconvs)) if tconvs else np.nan,
    }

# --- Main Evaluation Loop ---
