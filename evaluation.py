import numpy as np
import warnings
import os, json, hashlib
from simulation import sim_closed_loop
from sampling import sample_conditions

def rollout_success(config, controller, x0, eps=1e-2):
    # Suppress specific warnings during simulation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Suppress numpy specific errors (overflow in exp, square, etc.)
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
            # Restore previous numpy error settings
            np.seterr(**old_settings)

    if status < 0 or np.isnan(X).any(): 
        return False, np.inf
    norms = config.norm(X)
    idx = np.where(norms <= eps)[0]
    if len(idx) == 0:
        return False, np.inf
    tconv = t[idx[0]]
    return True, float(tconv)

def results_to_latex_table(
    results,
    *,
    caption: str | None = None,
    label: str | None = None,
    col_order=("Controller", "Dataset", "Stability (S)", "t_conv (med)", "Avg Cost (J)"),
    align="llrrr",
    float_format=None,          # optional callable(col, val)->str
    escape: bool = True,
):
    """
    Convert `evaluate_controller(...)` output (list of dicts) into a LaTeX table string.

    Requires: pandas.
    Produces: \\begin{table} ... \\begin{tabular} with booktabs.
    """
    import pandas as pd

    df = pd.DataFrame(results)
    df = df[[c for c in col_order if c in df.columns]]

    def _esc(s: str) -> str:
        # minimal LaTeX escaping
        return (s.replace("\\", "\\textbackslash{}")
                 .replace("&", "\\&")
                 .replace("%", "\\%")
                 .replace("$", "\\$")
                 .replace("#", "\\#")
                 .replace("_", "\\_")
                 .replace("{", "\\{")
                 .replace("}", "\\}")
                 .replace("~", "\\textasciitilde{}")
                 .replace("^", "\\textasciicircum{}"))

    if escape:
        for c in df.columns:
            df[c] = df[c].astype(str).map(_esc)

    # If you ever switch results to numeric types, you can pass float_format to format them here.

    tab = df.to_latex(
        index=False,
        escape=False,            # we already escaped (or not) above
        column_format=align,
        longtable=False,
        caption=None,
        label=None,
        bold_rows=False,
        na_rep="",
        multicolumn=False,
        multicolumn_format="c",
        float_format=None,
        booktabs=True,
    )

    if caption or label:
        cap = f"\\caption{{{_esc(caption) if escape else caption}}}\n" if caption else ""
        lab = f"\\label{{{_esc(label) if escape else label}}}\n" if label else ""
        tab = "\\begin{table}[t]\n\\centering\n" + cap + lab + tab + "\\end{table}\n"

    return tab

def evaluate_controller(config, controller, name, X_std, X_hard):
    results = []
    
    for dataset_name, X0_pool in [("Standard", X_std), ("Hard", X_hard)]:
        if X0_pool.shape[1] == 0: continue
        
        # 1. Run Stability & Convergence Stats
        stats = stability_score(config, controller, X0_pool=X0_pool)
        
        # 2. Compute Optimality (Average Trajectory Cost)
        total_costs = []
        n_eval = min(X0_pool.shape[1], 100) 
        
        # Suppress warnings during cost-based rollouts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in range(n_eval):
                t, X, status = sim_closed_loop(config.ocp.dynamics, config.ocp.closed_loop_jacobian,
                                               controller, tspan=[0.0, 10.0], X0=X0_pool[:, i])
                if status >= 0 and not np.isnan(X).any():
                    # Compute control inputs across the trajectory
                    u = controller.eval_U(X) 
                    # Calculate running cost
                    L = np.asarray(config.ocp.running_cost(X, u)).reshape(-1)
                    cost = np.trapz(L, t)
                    total_costs.append(cost)
        
        avg_cost = np.nanmean(total_costs) if total_costs else np.inf
        
        results.append({
            "Controller": name,
            "Dataset": dataset_name,
            "Stability (S)": f"{stats['S']:.1%}",
            "t_conv (med)": f"{stats['tconv_med']:.3f}s",
            "Avg Cost (J)": f"{avg_cost:.2e}"
        })
        

    return results

def _hard_ic_key(config, *, n_pool, eps, dist, seed):
    payload = dict(
        system=config.system,
        n_states=int(config.n_states),
        n_controls=int(config.n_controls),
        seed_cfg=int(getattr(config, "seed", -1)),
        # IC-generation params:
        n_pool=int(n_pool),
        eps=float(eps),
        dist=None if dist is None else float(dist),
        seed_sample=int(seed),
    )
    s = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(s).hexdigest(), payload

def ic_hard(config, *, n_pool=10000, eps=1e-2, dist=None, seed=0,
            cache_dir="./cache_hard_ics", force_regen=False):
    key, meta = _hard_ic_key(config, n_pool=n_pool, eps=eps, dist=dist, seed=seed)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"hard_ics_{key}.npz")

    if (not force_regen) and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["X_hard"], json.loads(str(z["__meta__"]))

    # sample pool and test under LQR on identical ICs
    X0 = sample_conditions(config, n=n_pool, seed=seed, dist=dist)  # (d,n_pool)
    lqr = config.ocp.LQR

    hard = []
    for i in range(n_pool):
        ok, _tconv = rollout_success(config, lqr, X0[:, i])
        if not ok:
            hard.append(X0[:, i])

    X_hard = np.array(hard).T if hard else np.zeros((config.n_states, 0))
    meta_out = {**meta, "n_hard": int(X_hard.shape[1])}

    np.savez_compressed(path, X_hard=X_hard, __meta__=json.dumps(meta))
    return X_hard, meta

import pandas as pd
import os, json, hashlib
from simulation import sim_closed_loop
from sampling import sample_conditions

def rollout_success(config, controller, x0, eps=1e-2):
    # This suppresses the terminal output for this specific simulation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Also catch numpy-specific errors like exp overflow
        old_settings = np.seterr(over="ignore", invalid="ignore", divide="ignore")
        
        try:
            t, X, status = sim_closed_loop(
                config.ocp.dynamics,
                config.ocp.closed_loop_jacobian,
                controller,
                tspan=[0.0, config.t1_initial],
                X0=np.asarray(x0, dtype=float),
                solver="LSODA",
                atol=1e-06,
                rtol=1e-03,
                events=config.ocp.make_integration_events(),
            )
        finally:
            np.seterr(**old_settings)

    if status < 0 or np.isnan(X).any():
        return False, np.inf

    norms = config.norm(X)
    idx = np.where(norms <= eps)[0]
    return (len(idx) > 0), (float(t[idx[0]]) if len(idx) > 0 else np.inf)



def stability_score(config, controller, n=500, eps=1e-2, dist=None, seed=0, X0_pool=None):
    if X0_pool is None:
        X0_pool = sample_conditions(config, n=n, dist=dist, seed=seed)   # your “basic” distribution
    else:
        X0_pool = np.asarray(X0_pool)
        n = X0_pool.shape[1]
    ok = 0
    tconvs = []
    for i in range(n):
        success, tconv = rollout_success(config, controller, X0_pool[:, i], eps=eps)
        ok += int(success)
        if success: tconvs.append(tconv)
    return {
        "S": ok / n,
        "tconv_med": float(np.median(tconvs)) if tconvs else np.nan,
        "tconv_p90": float(np.quantile(tconvs, 0.9)) if tconvs else np.nan,
    }









