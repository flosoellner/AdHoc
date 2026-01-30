"""
figures.py: plotting and tables in one module. Shared output path logic.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, Markdown

from sampling import sample_conditions
from simulation import sim_closed_loop


# -------------------------------------------------------------------------
# Shared: single helper for experiments/results/{system}/seed_{seed}/{subdir}/filename
# -------------------------------------------------------------------------

def _output_path(savepath, config=None, subdir="plots"):
    """
    If config has system/seed, resolve savepath to experiments/results/{system}/seed_{seed}/{subdir}/filename.
    Otherwise return savepath unchanged. subdir is "plots" or "tables".
    """
    if savepath is None:
        return None
    p = Path(savepath)
    if p.is_absolute():
        return p if subdir == "plots" else str(p)
    if config and hasattr(config, "system") and hasattr(config, "seed"):
        from problems import get_results_dir
        return str(Path(get_results_dir(config, subdir)) / p.name)
    return p if subdir == "plots" else savepath


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

# ---------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------

def _infer_system(config=None) -> str | None:
    if config is not None and hasattr(config, "system"):
        s = getattr(config, "system")
        return None if s is None else str(s)
    try:
        from problems import default_system
        return default_system
    except Exception:
        return None


def _iter_named_controllers(controllers):
    """
    Normalize a controllers collection into an iterator of (name, controller).

    Accepts:
      - list/tuple of (name, controller) OR (controller, name)
      - dict {name: controller}
      - list of controller objects (names auto-derived)
    """
    if isinstance(controllers, dict):
        items = list(controllers.items())
    else:
        items = list(controllers)

    def _looks_like_controller(obj) -> bool:
        return (
            hasattr(obj, "eval_U")
            or hasattr(obj, "eval_V")
            or hasattr(obj, "eval_dVdX")
            or hasattr(obj, "grad_net")
        )

    for item in items:
        # bare controller object
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            ctrl = item
            name = getattr(ctrl, "__name__", ctrl.__class__.__name__)
            yield str(name), ctrl
            continue

        a, b = item

        # Prefer a string as the name.
        if isinstance(a, str) and not isinstance(b, str):
            yield a, b
            continue
        if isinstance(b, str) and not isinstance(a, str):
            yield b, a
            continue

        # Otherwise choose the side that "looks like a controller".
        a_is_ctrl = _looks_like_controller(a)
        b_is_ctrl = _looks_like_controller(b)
        if a_is_ctrl and not b_is_ctrl:
            yield str(b), a
        elif b_is_ctrl and not a_is_ctrl:
            yield str(a), b
        else:
            # fall back: treat first as name-like, second as controller-like
            yield str(a), b


def _is_lqr(name, obj) -> bool:
    # Kept for backward compatibility (no longer used for styling).
    if "lqr" in str(name).lower():
        return True
    return obj.__class__.__name__.lower() == "lqr"

cmap = LinearSegmentedColormap.from_list(
    "blue_purple_blend",
    ["#56B4E9", "#7B2CBF"],
    N=256,
)


def _save_fig(fig, savepath, *, dpi=300):
    if savepath is None:
        return
    p = Path(savepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=dpi)

def V_from_gradnet_line_integral(grad_net, X, *, n_quad=16):
    """
    Reconstruct scalar V(x) from a GradNet g(x) ≈ ∇V(x) via
        V(x) = ∫_0^1 <g(αx), x> dα   with V(0)=0.

    Args:
        grad_net: torch nn.Module mapping (N,d)->(N,d)
        X: torch tensor (N,d)
        n_quad: number of quadrature points on [0,1]

    Returns:
        V: torch tensor (N,)
    """
    alphas = torch.linspace(0.0, 1.0, int(n_quad), device=X.device, dtype=X.dtype)  # (Q,)
    Xa = alphas[:, None, None] * X[None, :, :]                                      # (Q,N,d)
    g = grad_net(Xa.reshape(-1, X.shape[1])).reshape(int(n_quad), X.shape[0], X.shape[1])  # (Q,N,d)
    ip = (g * X[None, :, :]).sum(dim=2)                                             # (Q,N)
    V = torch.trapz(ip, alphas, dim=0)                                              # (N,)
    return V


def _value_from_controller_or_gradnet(controller, X, *, device="cpu", n_quad=64):
    """
    X: (d,N) numpy
    Returns V: (N,) numpy
    """
    # direct value available (e.g. LQR has eval_V)
    if hasattr(controller, "eval_V"):
        return np.asarray(controller.eval_V(X)).reshape(-1)

    # reconstruct from grad_net via line integral
    if hasattr(controller, "grad_net") and (controller.grad_net is not None):
        controller.grad_net.eval()
        Xt = torch.tensor(X.T, dtype=torch.float32, device=device)  # (N,d)
        with torch.no_grad():
            Vt = V_from_gradnet_line_integral(controller.grad_net, Xt, n_quad=int(n_quad))
        return Vt.detach().cpu().numpy().reshape(-1)

    raise ValueError("Controller must provide eval_V(X) or have .grad_net for reconstruction.")



def plot_value_vs_state_norm(
    *,
    config,
    controllers,                 # list like [("LQR", config.ocp.LQR), ("ctrl1", ctrl1)]
    n=1000,
    seed=0,
    dist=None,
    n_quad=32,
    device="cpu",
    alpha=0.25,
    s=6,
    title=None,
    figsize=(6.2, 3.2),
    savepath=None,
):
    """
    Samples x ~ sample_conditions, then plots ||x||_w (x-axis) vs V(x) (y-axis).
    """
    X = sample_conditions(config, n=int(n), seed=int(seed), dist=dist)  # (d,N)
    xnorm = np.asarray(config.ocp.norm(X)).reshape(-1)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for k, (name, ctrl) in enumerate(_iter_named_controllers(controllers)):
        V = _value_from_controller_or_gradnet(ctrl, X, device=device, n_quad=n_quad)



        # in plot_value_vs_state_norm:
        ax.scatter(xnorm, V, s=float(s), alpha=float(alpha), label=str(name))

    # hide negative values by cutting y-axis at 0
    _ymin, _ymax = ax.get_ylim()
    ax.set_ylim(0.0, max(0.0, float(_ymax)))

    ax.set_xlabel(r"$\|x\|_w$")
    ax.set_ylabel(r"$V(x)$")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save_fig(fig, _output_path(savepath, config, "plots"))
    return fig

class _ZeroController:
    def __init__(self, config):
        self.m = int(config.n_controls)

    def eval_U(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            return np.zeros((self.m,), dtype=float)
        return np.zeros((self.m, X.shape[1]), dtype=float)


def plot_3d(
    *,
    config,
    controller,
    tspan=None,
    Nt=250,
    seed=0,
    dist=None,
    solver="LSODA",
    events="auto",
    figsize=(6.8, 3.6),
    elev=25,
    azim=-60,
    cmap=cmap,
    savepath=None,
):
    if savepath is None and config is not None and hasattr(config, "system"):
        savepath = f"figures/{config.system}_3d.pdf"
    # 1) sample one IC
    X0 = sample_conditions(config, n=1, seed=seed, dist=dist)[:, 0]

    # 2) Use t1_initial from config if tspan not provided
    if tspan is None:
        tspan = (0.0, config.t1_initial)
    
    # 3) common time grid
    t_eval = np.linspace(float(tspan[0]), float(tspan[1]), int(Nt))

    # 4) events
    if events == "auto":
        events = config.ocp.make_integration_events()

    # 5) simulate
    t_u, X_u, _ = sim_closed_loop(
        config.ocp.dynamics, config.ocp.closed_loop_jacobian,
        _ZeroController(config),
        tspan=list(tspan), X0=X0, t_eval=t_eval, events=events, solver=solver
    )
    t_c, X_c, _ = sim_closed_loop(
        config.ocp.dynamics, config.ocp.closed_loop_jacobian,
        controller,
        tspan=list(tspan), X0=X0, t_eval=t_eval, events=events, solver=solver
    )



    # 6) plot
# 6) plot
    # Check if we are in a state-space system (Finance/VdP) or a PDE system
    if config.xi is None:
        # For Finance, we treat the 'Asset Index' as the y-axis
        xi = np.arange(config.n_states) 
    else:
        xi = np.asarray(config.xi).reshape(-1)

    # Ensure X_u and X_c are shaped as (n_states, n_time_steps)
    if X_u.shape[0] != xi.size:
        # Fallback for state-space systems if xi was misconfigured
        xi = np.arange(X_u.shape[0])

    Tg, Xig = np.meshgrid(t_eval, xi)

    # ... rest of the plotting code ...

    # --- create figure/axes (NO constrained_layout for 3D) ---
    fig = plt.figure(figsize=(7.8, 3.2), constrained_layout=False)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # symmetric margins → centered
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.90, wspace=0.35)

    # --- plot surfaces as before ---
    ax1.plot_surface(Tg, Xig, X_u, cmap=cmap, linewidth=0, antialiased=True)
    ax2.plot_surface(Tg, Xig, X_c, cmap=cmap, linewidth=0, antialiased=True)

    # --- “subcaptions” closer to the plots ---
    ax1.set_title(r"$\bm{u}\equiv 0$", pad=-6)
    ax2.set_title(r"$\bm{u}\equiv\bm{u}^*$", pad=-6)

    zmax = float(np.nanmax(np.abs(np.concatenate([X_u.ravel(), X_c.ravel()]))))
    if not np.isfinite(zmax) or zmax <= 0:
        zmax = 1.0  # fallback

    ax1.set_zlim(-zmax, zmax)
    ax2.set_zlim(-zmax, zmax)

    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.xaxis.pane.set_facecolor((1, 1, 1, 1))
        ax.yaxis.pane.set_facecolor((1, 1, 1, 1))
        ax.zaxis.pane.set_facecolor((1, 1, 1, 1))
        ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))
        ax.set_xlabel(r"$t$", labelpad=-12)
        ax.set_ylabel(r"$\xi$", labelpad=-12)
        ax.set_zlabel(r"$x(t,\xi)$", labelpad=-12)
        ax.tick_params(pad=0)
        ax.grid(True)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["linewidth"] = 0.4
            axis._axinfo["grid"]["color"] = (0, 0, 0, 0.15)  # transparent black
    for ax in (ax1, ax2):
        # y gridlines at each xi gridpoint
        ax.set_yticks(xi.tolist())

        # hide x gridlines (keep y/z)
        ax.xaxis._axinfo["grid"]["color"] = (0, 0, 0, 0.0)

        # hide numeric tick labels (keep axes & ticks)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # optional: also reduce tick marks themselves
        ax.tick_params(length=0)

    _save_fig(fig, _output_path(savepath, config, "plots"))

    return fig




def _eval_grad(controller, X: np.ndarray) -> np.ndarray:
    """
    Return dVdX as (d,N) for a controller.
    Supports:
      - LQR-like: controller.eval_dVdX(X)
      - NN controller wrapper: controller.grad_net (torch model)
    """
    if hasattr(controller, "eval_dVdX"):
        return np.asarray(controller.eval_dVdX(X))
    if hasattr(controller, "grad_net") and (controller.grad_net is not None):
        x_t = torch.tensor(X.T, dtype=torch.float32)
        with torch.no_grad():
            g = controller.grad_net(x_t).cpu().numpy().T
        return g
    raise ValueError("Controller must provide eval_dVdX(X) or have .grad_net")





def plot_value_analysis_combined(
    *,
    config,
    controllers,
    # Parameters for plot_value_vs_state_norm
    n=1000,
    seed=0,
    dist=None,
    n_quad=32,
    device="cpu",
    alpha=0.25,
    s=6,
    # Parameters for plot_value_flow
    x0=None,
    tspan=(0.0, 1.0),
    Nt=400,
    solver="LSODA",
    events="auto",
    bump_tol=0.0,
    mark_bumps=True,
    # Parameters for plot_hjb_residual_shock_line
    shock=None,
    shock_seed=0,
    s_grid=None,
    dist_for_shock=1.0,
    metric="abs",
    log10=False,
    eps=1e-12,
    # Combined parameters
    title=None,
    figsize=(6.2, 9.6),  # 3 plots stacked vertically
    savepath=None,
    suffix: str = "",    # used for default savepath: figures/{system}_value_analysis{suffix}.pdf
):
    if savepath is None and config is not None and hasattr(config, "system"):
        savepath = f"figures/{config.system}_value_analysis{suffix}.pdf"
    """
    Creates three separate plots and saves each as a PNG file:
    1. Value vs state norm (scatter)
    2. Value flow along trajectory (line)
    3. HJB residual shock line (line)
    
    Ensures consistent colors across all three plots for each controller.
    If savepath is provided, generates three PNG files with suffixes:
    - {savepath}_value_vs_norm.png
    - {savepath}_value_flow.png
    - {savepath}_hjb_residual.png
    """
    from controls.train import loss_unified
    
    # Normalize controllers to list of (name, controller) tuples
    controller_list = list(_iter_named_controllers(controllers))
    n_controllers = len(controller_list)
    
    # Create consistent color mapping using matplotlib's default cycle
    # Get the default color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_list = prop_cycle.by_key()['color']
    # Extend if needed
    while len(colors_list) < n_controllers:
        colors_list.extend(colors_list)
    
    # Create color map: name -> color
    color_map = {name: colors_list[k] for k, (name, _) in enumerate(controller_list)}
    
    # Generate savepaths for 3 separate PNG files
    def _make_savepath(base_path, suffix):
        if base_path is None:
            return None
        p = Path(base_path)
        # Replace extension with .png and add suffix
        stem = p.stem
        parent = p.parent
        return str(parent / f"{stem}_{suffix}.png")
    
    savepath1 = _make_savepath(savepath, "value_vs_norm")
    savepath2 = _make_savepath(savepath, "value_flow")
    savepath3 = _make_savepath(savepath, "hjb_residual")
    
    # Individual figure size for each plot
    individual_figsize = (6.2, 3.2)
    
    # ============================================================
    # Plot 1: Value vs State Norm
    # ============================================================
    X = sample_conditions(config, n=int(n), seed=int(seed), dist=dist)  # (d,N)
    xnorm = np.asarray(config.ocp.norm(X)).reshape(-1)
    
    fig1, ax1 = plt.subplots(figsize=individual_figsize, constrained_layout=True)
    for name, ctrl in controller_list:
        V = _value_from_controller_or_gradnet(ctrl, X, device=device, n_quad=n_quad)
        color = color_map[name]
        ax1.scatter(xnorm, V, s=float(s), alpha=float(alpha), label=str(name), color=color)
    
    _ymin, _ymax = ax1.get_ylim()
    ax1.set_ylim(0.0, max(0.0, float(_ymax)))
    ax1.set_xlabel(r"$\|x\|_w$")
    ax1.set_ylabel(r"$V(x)$")
    ax1.set_title("Value vs State Norm")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    _save_fig(fig1, _output_path(savepath1, config, "plots"))
    
    # ============================================================
    # Plot 2: Value Flow
    # ============================================================
    if x0 is None:
        X0 = sample_conditions(config, n=1, seed=int(seed), dist=dist)  # (d,1)
        x0 = X0[:, 0]
    x0 = np.asarray(x0).reshape(-1)
    
    t_eval = np.linspace(float(tspan[0]), float(tspan[1]), int(Nt))
    
    if events == "auto":
        if hasattr(config, "ocp") and hasattr(config.ocp, "make_integration_events"):
            events = config.ocp.make_integration_events()
        else:
            events = None
    
    fig2, ax2 = plt.subplots(figsize=individual_figsize, constrained_layout=True)
    for name, ctrl in controller_list:
        color = color_map[name]
        t, X, status = sim_closed_loop(
            config.ocp.dynamics,
            config.ocp.closed_loop_jacobian,
            ctrl,
            tspan=list(tspan),
            X0=x0,
            t_eval=t_eval,
            events=events,
            solver=solver,
        )  # X: (d, Nt_eff)
        
        V = _value_from_controller_or_gradnet(ctrl, X, device=device, n_quad=int(n_quad))
        ax2.plot(t, V, label=str(name), color=color)
        
        if mark_bumps and V.size >= 2:
            dV = np.diff(V)
            bump_idx = np.where(dV > float(bump_tol))[0] + 1
            if bump_idx.size > 0:
                ax2.plot(t[bump_idx], V[bump_idx], "o", ms=3.5, alpha=0.9, color=color)
    
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$V(x(t))$")
    ax2.set_title("Value Flow")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    _save_fig(fig2, _output_path(savepath2, config, "plots"))
    
    # ============================================================
    # Plot 3: HJB Residual Shock Line
    # ============================================================
    if s_grid is None:
        s_grid = np.linspace(0.0, 2.0, 20)
    s_grid = np.asarray(s_grid, dtype=float).reshape(-1)
    
    d = int(config.n_states)
    
    if shock is None:
        Xs = sample_conditions(config, n=1, seed=int(shock_seed), dist=float(dist_for_shock))  # (d,1)
        shock = Xs[:, 0]
    shock = np.asarray(shock, dtype=float).reshape(-1)
    if shock.size != d:
        raise ValueError(f"shock must have shape ({d},)")
    
    class _LQRAsGradModel:
        """Shim so train.py loss can call model(X)->dVdX and access model.config."""
        def __init__(self, config):
            self.config = config
        
        def __call__(self, X):  # X: (B,d) torch
            G = self.config.ocp.LQR.eval_dVdX(X.detach().cpu().numpy().T)  # (d,B) numpy
            return torch.tensor(G.T, dtype=X.dtype, device=X.device)       # (B,d)
    
    def _as_grad_model(obj):
        if hasattr(obj, "eval_dVdX"):
            return _LQRAsGradModel(config)
        if hasattr(obj, "grad_net") and (obj.grad_net is not None):
            obj = obj.grad_net
        if not hasattr(obj, "config"):
            raise ValueError(
                "For non-LQR, pass the GradNet itself (e.g. ctrl.grad_net or model), "
                "not a string or a controller without gradient information."
            )
        return obj
    
    fig3, ax3 = plt.subplots(figsize=individual_figsize, constrained_layout=True)
    for name, obj in controller_list:
        color = color_map[name]
        model = _as_grad_model(obj)
        ys = []
        for s in s_grid:
            Xnp = shock[:, None] * float(s)                               # (d,1)
            Xt = torch.tensor(Xnp.T, dtype=torch.float32, device=device)  # (1,d)
            
            total, hjb_err, dpc_err, sup_err = loss_unified(
                model,
                (Xt,),
                mode="unsupervised",
                supervision=False,
            )
            
            hjb_sq = hjb_err
            val = float(hjb_sq.detach().cpu().numpy())
            
            if metric == "abs":
                val = np.sqrt(val + float(eps))
            elif metric == "sq":
                pass
            else:
                raise ValueError("metric must be 'abs' or 'sq'")
            
            if log10:
                val = np.log10(val + float(eps))
            
            ys.append(val)
        
        ax3.plot(s_grid, ys, label=str(name), color=color)
    
    ax3.set_xlabel(r"shock intensity $s$ (state $x = s\,x_{\mathrm{shock}}$)")
    ylab = (r"$\log_{10}(|\mathcal{H}|)$" if (metric == "abs" and log10) else
            r"$\log_{10}(|\mathcal{H}|^2)$" if (metric == "sq" and log10) else
            r"$|\mathcal{H}|$" if metric == "abs" else
            r"$|\mathcal{H}|^2$")
    ax3.set_ylabel(ylab)
    ax3.set_title("HJB Residual")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    _save_fig(fig3, _output_path(savepath3, config, "plots"))
    
    return fig1, fig2, fig3



def plot_space_time_heatmap(
    *,
    config,
    controller,                 # controller with eval_U(X)
    x0=None,                     # (d,) or (d,1); if None, sampled
    seed: int = 0,
    dist=None,
    tspan=(0.0, 10.0),
    Nt: int = 350,
    solver="LSODA",
    events="auto",
    cmap="BuPu",
    symmetric: bool = True,      # symmetric color limits around 0
    vlim=None,                   # tuple (vmin,vmax) overrides symmetric
    title: str | None = None,
    figsize=(6.8, 3.2),
    savepath=None,
):
    """Space-time heatmap of the state profile x(t,xi)."""

    ctrl = _ZeroController(config) if (controller is None) else controller

    # x0
    if x0 is None:
        X0 = sample_conditions(config, n=1, seed=int(seed), dist=dist)  # (d,1)
        x0 = X0[:, 0]
    x0 = np.asarray(x0, dtype=float).reshape(-1)

    # time grid
    t_eval = np.linspace(float(tspan[0]), float(tspan[1]), int(Nt))

    # events
    if events == "auto":
        events = config.ocp.make_integration_events() if hasattr(config.ocp, "make_integration_events") else None

    # simulate
    t, X, status = sim_closed_loop(
        config.ocp.dynamics,
        config.ocp.closed_loop_jacobian,
        ctrl,
        tspan=list(tspan),
        X0=x0,
        t_eval=t_eval,
        events=events,
        solver=solver,
    )  # X: (d, Nt_eff)

    xi = np.asarray(config.xi).reshape(-1)
    if X.shape[0] != xi.size:
        raise ValueError(f"Mismatch: X has {X.shape[0]} states but xi has {xi.size} points")

    # color limits
    if vlim is not None:
        vmin, vmax = float(vlim[0]), float(vlim[1])
    elif symmetric:
        m = float(np.nanmax(np.abs(X)))
        if not np.isfinite(m) or m <= 0:
            m = 1.0
        vmin, vmax = -m, m
    else:
        vmin = vmax = None

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(
        X,
        origin="lower",
        aspect="auto",
        extent=[t[0], t[-1], xi[0], xi[-1]],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\xi$")
    ax.set_title(title or "Space–time heatmap $x(t,\\xi)$")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$x(t,\xi)$")

    _save_fig(fig, _output_path(savepath, config, "plots"))
    return fig

def plot_gradient_attention_heatmap(
    *,
    config,
    controller,                 # LQR / NN ctrl; must provide eval_dVdX(X) or .grad_net
    x0=None,                     # (d,) or (d,1); if None, sampled
    seed: int = 0,
    dist=None,
    tspan=(0.0, 30.0),
    Nt: int = 350,
    solver="LSODA",
    events="auto",
    device="cpu",
    weight_norm: bool = False,   # if True, plot sqrt(w_i)*|dVdX_i| (weighted)
    log10: bool = False,
    eps: float = 1e-12,
    cmap="BuPu",
    title: str | None = None,
    figsize=(6.8, 3.2),
    savepath=None,
):
    """
    "Attention" heatmap: time vs space, color = |∂V/∂x_i| along a trajectory x(t).

    X-axis: t
    Y-axis: xi (spatial grid)
    Color: |dVdX_i(t)|
    """
    # x0
    if x0 is None:
        X0 = sample_conditions(config, n=1, seed=int(seed), dist=dist)  # (d,1)
        x0 = X0[:, 0]
    x0 = np.asarray(x0, dtype=float).reshape(-1)

    # time grid
    t_eval = np.linspace(float(tspan[0]), float(tspan[1]), int(Nt))

    # events
    if events == "auto":
        events = config.ocp.make_integration_events() if hasattr(config.ocp, "make_integration_events") else None

    # simulate
    t, X, status = sim_closed_loop(
        config.ocp.dynamics,
        config.ocp.closed_loop_jacobian,
        controller,
        tspan=list(tspan),
        X0=x0,
        t_eval=t_eval,
        events=events,
        solver=solver,
    )  # X: (d, Nt_eff)

    xi = np.asarray(config.xi).reshape(-1)
    if X.shape[0] != xi.size:
        raise ValueError(f"Mismatch: X has {X.shape[0]} states but xi has {xi.size} points")

    # gradients along trajectory: need X as (d,Nt)
    G = _eval_grad(controller, X)             # (d, Nt)
    A = np.abs(G)                            # (d, Nt)

    if weight_norm:
        w = getattr(getattr(config, "ocp", None), "w_flat", None)
        if w is not None:
            w = np.sqrt(np.asarray(w).reshape(-1))
            A = (w[:, None] * A)

    if log10:
        A = np.log10(A + float(eps))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(
        A,
        origin="lower",
        aspect="auto",
        extent=[t[0], t[-1], xi[0], xi[-1]],
        cmap=cmap,
        interpolation="nearest",
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\xi$")
    lab = (r"$\log_{10}|\partial V/\partial x(\xi)|$" if log10 else r"$|\partial V/\partial x(\xi)|$")
    ax.set_title(title or ("Gradient attention heatmap: " + lab))
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(lab)

    _save_fig(fig, _output_path(savepath, config, "plots"))
    return fig

def plot_training_losses(
    *,
    iters=None,
    losses=None,
    label: str = "loss",
    logy: bool = True,
    color: str | None = None,
    title: str | None = None,
    xlabel: str = "iteration",
    ylabel: str = "loss",
    figsize=(6.2, 3.2),
    savepath=None,
    config=None,
    plot_name: str = "loss_curve",   # used for default savepath: figures/{system}_{plot_name}.pdf
    # NEW: multi-series support (preferred)
    series=None,  # list of (name, iters, losses)
    # NEW:
    smooth: str | None = "ema",     # None | "ema" | "ma"
    ema_alpha: float = 0.03,        # for EMA
    ma_window: int = 200,           # for moving average
):
    if savepath is None and config is not None and hasattr(config, "system"):
        savepath = f"figures/{config.system}_{plot_name}.pdf"
    # Allow passing history dicts: (name, history_dict) or (name, iters, losses, phase)
    norm_series = []
    for item in series:
        if len(item) == 2 and isinstance(item[1], dict):
            name_k, h = item
            norm_series.append((name_k, h.get("iters"), h.get("loss"), h.get("phase")))
        elif len(item) == 3:
            name_k, it_k, lo_k = item
            norm_series.append((name_k, it_k, lo_k, None))
        elif len(item) == 4:
            name_k, it_k, lo_k, ph_k = item
            norm_series.append((name_k, it_k, lo_k, ph_k))
        else:
            raise ValueError("Each series entry must be (name,iters,losses), (name,history_dict), or (name,iters,losses,phase).")

    # Extract unsupervised phases and reset iterations - SAME for ALL controllers
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for k, (name_k, it_k, lo_k, ph_k) in enumerate(norm_series):
        if it_k is None or lo_k is None:
            continue
            
        it_k = np.asarray(it_k).reshape(-1)
        lo_k = np.asarray(lo_k).reshape(-1)

        # For ALL controllers: extract unsupervised phase (if phase info exists)
        if ph_k is not None:
            ph = np.asarray(ph_k).reshape(-1)
            if ph.size == it_k.size:
                unsup_mask = (ph != "sup")
                if not np.any(unsup_mask):
                    continue  # Skip if all supervised
                it_k = it_k[unsup_mask]
                lo_k = lo_k[unsup_mask]
        elif "hybrid" in str(name_k).lower():
            # For hybrid models without phase info (old checkpoints):
            # Detect phase boundary by looking for a jump in iterations
            # Supervised phase has consecutive iterations, then unsupervised starts at higher number
            if len(it_k) > 1:
                diffs = np.diff(it_k)
                # Find large jump (likely boundary between sup and unsup)
                large_jump_idx = np.where(diffs > np.percentile(diffs, 90))[0]
                if len(large_jump_idx) > 0:
                    # Assume everything after the first large jump is unsupervised
                    unsup_start = large_jump_idx[0] + 1
                    it_k = it_k[unsup_start:]
                    lo_k = lo_k[unsup_start:]
        
        # For ALL controllers: reset iterations to start at 1
        if len(it_k) == 0:
            continue
        it_k = it_k - it_k[0] + 1

        # Smooth and plot - SAME for ALL
        if smooth is None:
            it_plot, y_plot = it_k, lo_k
        elif smooth == "ema":
            y = np.empty_like(lo_k, dtype=float)
            y[0] = float(lo_k[0])
            a = float(ema_alpha)
            for i in range(1, len(lo_k)):
                y[i] = a * float(lo_k[i]) + (1.0 - a) * y[i - 1]
            it_plot, y_plot = it_k, y
        elif smooth == "ma":
            w = int(ma_window)
            if w < 2 or w > len(lo_k):
                raise ValueError("ma_window must be in [2, len(losses)]")
            y = np.convolve(lo_k, np.ones(w) / w, mode="valid")
            it_plot = it_k[w - 1 :]
            y_plot = y
        else:
            raise ValueError("smooth must be None, 'ema', or 'ma'")

        if color is not None:
            ax.plot(it_plot, y_plot, color=str(color), lw=1.8, label=str(name_k))
        else:
            ax.plot(it_plot, y_plot, label=str(name_k))

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save_fig(fig, _output_path(savepath, config, "plots"))
    return fig, ax

# -------------------------------------------------------------------------
# Tables
# -------------------------------------------------------------------------

def eikonal_eval_series(ocp, controller, X_val=None, dVdX_val=None):
    """
    Eikonal evaluation metrics as a pandas Series.
    HJB residual always; MSE vs exact only when dim_d == 1.
    """
    from evaluation import evaluate_eikonal
    m = evaluate_eikonal(ocp, controller, X_val=X_val, dVdX_val=dVdX_val)
    row = {"HJB res (mean)": m["hjb_residual_mean"], "HJB res (max)": m["hjb_residual_max"]}
    if m["mse_grad"] is not None:
        row["MSE ∇V"] = m["mse_grad"]
    if m["mse_V"] is not None:
        row["MSE V"] = m["mse_V"]
    return pd.Series(row)

def save_dataframe_latex(df, savepath, caption=None, label=None, precision=4):
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    disp_df = df.copy()
    
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.2e}" if (abs(x) < 1e-4 or abs(x) > 1e4) else f"{x:.{precision}g}"
        return str(x)

    def escape_tex(s):
        return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
        
    for col in disp_df.columns:
        disp_df[col] = disp_df[col].map(fmt).map(escape_tex)
    disp_df.columns = disp_df.columns.map(escape_tex)

    # Use Styler for Pandas 2.0+
    styler = disp_df.style.hide(axis="index")
    latex_str = styler.to_latex(
        column_format="l" + "r" * (len(df.columns) - 1),
        hrules=True, label=label, caption=caption,
        position="H", position_float="centering"
    )

    with open(savepath, "w", encoding="utf-8") as f:
        f.write(latex_str)

# --- Notebook Display Helper ---

def show_spec(obj, keys, title=None):
    """
    Cleaner replacement for show_katex_array.
    Creates a spec on the fly and displays it as a styled HTML table.
    """
    valid_keys = [k for k in keys if hasattr(obj, k)]
    vals = [getattr(obj, k) for k in valid_keys]
    
    df = pd.DataFrame([vals], columns=valid_keys)
    if title:
        display(Markdown(f"**{title}**"))
    
    # Stylish Jupyter display
    display(df.style.hide(axis="index").set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f4f4f4'), ('color', 'black'), ('font-weight', 'bold')]}
    ]))

# --- Specific Thesis Helpers ---

def save_config_table(config, savepath="config.tex"):
    """Save config table in horizontal format (one row of headers, one row of values)."""
    keys = ["system", "seed", "n_states", "n_controls", "t1_initial", "fp_tol"]
    valid_keys = [k for k in keys if hasattr(config, k)]
    values = [getattr(config, k) for k in valid_keys]
    
    # Format values
    formatted_values = []
    for k, v in zip(valid_keys, values):
        if k == "system":
            formatted_values.append(str(v))
        elif k == "seed":
            formatted_values.append(str(int(v)))
        elif k in ["n_states", "n_controls"]:
            formatted_values.append(str(int(v)))
        elif k == "t1_initial":
            formatted_values.append(f"{v:.1f}")
        elif k == "fp_tol":
            formatted_values.append(f"{v:.0e}")
        else:
            formatted_values.append(str(v))
    
    full_path = _output_path(savepath, config, "tables")
    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
    
    # Format header names
    header_map = {
        "system": "system",
        "seed": "seed",
        "n_states": "$n$",
        "n_controls": "$m$",
        "t1_initial": "$t_1$",
        "fp_tol": "$\\epsilon$"
    }
    headers = [header_map.get(k, k) for k in valid_keys]
    
    with open(full_path, "w", encoding="utf-8") as f:
        n_cols = len(valid_keys)
        f.write(f"\\begin{{tabular}}{{@{{}}{'c' * n_cols}@{{}}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        f.write(" & ".join(formatted_values) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

def save_results_table(results_list, config=None, savepath="results.tex"):
    """
    Reformats the flat results list into a professional Thesis table.
    Groups by Controller and stacks Standard/Hard datasets as rows.
    """
    import pandas as pd
    df = pd.DataFrame(results_list)
    
    # 1. Clean up the naming for the Thesis
    # Use LaTeX math mode for headers (e.g., $t_{conv}$)
    rename_map = {
        "Controller": "Method",
        "Dataset": "Data",
        "Stability (S)": "$S$",
        "Avg Cost (J)": "$J_{\text{avg}}$"
    }
    df = df.rename(columns=rename_map)

    # 2. Reorder columns to put Method and Data first
    cols = ["Method", "Data", "$S$", "$t_{\text{conv}}$", "$J_{\text{avg}}$"]
    df = df[[c for c in cols if c in df.columns]]

    # 3. Format 'S' as a proper percentage if it's a float
    if "$S$" in df.columns:
        df["$S$"] = df["$S$"].apply(lambda x: f"{x:.0%}" if isinstance(x, (float, int)) else x)

    # 4. Save using our fixed LaTeX function
    full_path = _output_path(savepath, config, "tables")
    save_dataframe_latex(
        df, 
        full_path, 
        caption="Performance comparison of learned controllers vs. LQR baseline.", 
        label="tab:results"
    )

def save_data_summary_table(config, data, savepath="data_summary.tex"):
    """Save data summary table in specific LaTeX format."""
    import numpy as np
    
    X_all = np.asarray(data["X"])
    t_all = np.asarray(data["t"]).reshape(-1)
    X_norms = config.norm(X_all).reshape(-1)
    
    n_traj = int(data.get("n_trajectories", -1))
    n_points = int(X_all.shape[1])
    t_min = t_all.min()
    t_max = t_all.max()
    abs_x_mean = np.mean(np.abs(X_all))
    abs_x_max = np.max(np.abs(X_all))
    norm_x_mean = np.mean(X_norms)
    norm_x_max = np.max(X_norms)
    
    full_path = _output_path(savepath, config, "tables")
    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
    
    with open(full_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{@{}cccccccc@{}}\n")
        f.write("\\toprule\n")
        f.write("$N_{\\mathrm{traj}}$ & $|\\mathcal{D}|$ & $t$ (min) & $t$ (max) & $|x|$ (mean) & $|x|$ (max)& $||x||$ (mean)& $||x||$ (max)\\\\\n")
        f.write("\\midrule\n")
        f.write(f"{n_traj} & {n_points} & {t_min:.2f} & {t_max:.2f} & {abs_x_mean:.2f} & {abs_x_max:.2f} & {norm_x_mean:.2f} & {norm_x_max:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    
    # Return a DataFrame for compatibility
    return pd.DataFrame({
        "$N_{\\mathrm{traj}}$": [n_traj],
        "$|\\mathcal{D}|$": [n_points],
        "t (min)": [t_min],
        "t (max)": [t_max],
        "|x| (mean)": [abs_x_mean],
        "|x| (max)": [abs_x_max],
        "||x|| (mean)": [norm_x_mean],
        "||x|| (max)": [norm_x_max]
    })

def save_params_table(obj, savepath, title="Configuration", keys=None, config=None):
    """Generic function to save any object's attributes to a LaTeX table."""
    if keys is None:
        keys = [k for k in obj.__dict__.keys() if not k.startswith('_')]
    
    data = {
        "Parameter": [str(k) for k in keys if hasattr(obj, k)],
        "Value": [getattr(obj, k) for k in keys if hasattr(obj, k)]
    }
    df = pd.DataFrame(data)
    
    full_path = _output_path(savepath, config, "tables")
    
    # Special handling for training config
    if "traincfg" in savepath.lower() or "train" in savepath.lower():
        _save_train_config_latex(obj, full_path)
    else:
        save_dataframe_latex(df, full_path, caption=title, label=f"tab:{Path(savepath).stem}")

def _save_train_config_latex(cfg, savepath):
    """Save training config in specific LaTeX format."""
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    
    def fmt_sci(x):
        """Format number as LaTeX scientific notation if small."""
        if isinstance(x, (int, float)):
            if x >= 1e-2:
                return f"{x:.2g}"
            exp = int(np.floor(np.log10(x)))
            coeff = x / (10 ** exp)
            if abs(coeff - int(coeff)) < 1e-6:
                return f"${int(coeff)}\\times 10^{{{exp}}}$"
            return f"${coeff:.1f}\\times 10^{{{exp}}}$"
        return str(x)
    
    epochs_sup = getattr(cfg, 'sup_epochs', 1)
    epochs_unsup = getattr(cfg, 'unsup_epochs', 5)
    steps = getattr(cfg, 'unsup_n_steps', 70)
    horizon = getattr(cfg, 'horizon', 30)
    lr = getattr(cfg, 'unsup_lr', 5e-4)
    lambda_sup = getattr(cfg, 'lambda_sup_base', 0.5)
    batch_size = getattr(cfg, 'batch_size', None)
    batch_size_str = str(int(batch_size)) if batch_size is not None else "None"
    
    with open(savepath, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{@{}ccccccc@{}}\n")
        f.write("\\toprule\n")
        f.write("$E_{\\mathrm{sup}}$ & $E_{\\mathrm{unsup}}$ & $S_{\\mathrm{unsup}}$ & $h$ & $\\lambda_{\\text{sup}}$ & $\\mu$ & $|\\mathcal{B}|$  \\\\\n")
        f.write("\\midrule\n")
        f.write(f"{epochs_sup} & {epochs_unsup} & {steps} & {horizon} & {lambda_sup} & {fmt_sci(lr)} & {batch_size_str}  \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def show_monte_carlo_results(results_dict, controller_name="Controller", title=None):
    """
    Display Monte Carlo evaluation results in a nice formatted table.
    
    Parameters
    ----------
    results_dict : dict or dict of dicts
        Output from simulation.monte_carlo() - can be single controller or multiple
    controller_name : str, optional
        Name if single controller (ignored if results_dict contains multiple)
    title : str, optional
        Title for the table
    """
    import numpy as np
    
    # Check if this is a dict of results (multiple controllers)
    if isinstance(results_dict, dict) and 'X0_pool' not in results_dict:
        # Multiple controllers - create comparison table
        rows = []
        for name, res in results_dict.items():
            rows.append(_compute_monte_carlo_stats(res, name))
        
        df = pd.DataFrame(rows)
    else:
        # Single controller
        df = _compute_monte_carlo_stats(results_dict, controller_name)
        df = pd.DataFrame([df])
    
    if title:
        display(Markdown(f"**{title}**"))
    
    display(df.style.hide(axis="index").set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f4f4f4'), ('color', 'black'), ('font-weight', 'bold')]}
    ]))
    
    return df

def _compute_monte_carlo_stats(results_dict, controller_name):
    """Helper to compute stats for a single controller."""
    import numpy as np
    
    init_dists = np.asarray(results_dict['init_dists'])
    final_dists = np.asarray(results_dict['final_dists'])
    NN_final_times = np.asarray(results_dict['NN_final_times'])
    NN_costs = np.asarray(results_dict['NN_costs'])
    
    converged_mask = np.isfinite(NN_final_times)
    n_converged = np.sum(converged_mask)
    n_total = len(NN_final_times)
    stability = n_converged / n_total if n_total > 0 else 0.0
    
    return {
        "Controller": controller_name,
        "Stability (S)": f"{stability:.1%}",
      #  "Converged / Total": f"{n_converged} / {n_total}",
      #  "Initial ||X|| (mean)": f"{np.mean(init_dists):.4f}",
        "Final ||X|| (mean)": f"{np.mean(final_dists[converged_mask]):.4f}" if n_converged > 0 else "N/A",
        "t_conv (mean)": f"{np.mean(NN_final_times[converged_mask]):.2f}" if n_converged > 0 else "N/A",
        "Cost J (mean)": f"{np.mean(NN_costs[converged_mask]):.4f}" if n_converged > 0 else "N/A"
    }


def save_monte_carlo_results(results_dict, controller_name="Controller", config=None, 
                            savepath="monte_carlo.tex"):
    """Save Monte Carlo results to LaTeX table with format: Model | Stability S | t_conv (mean) | Cost J (mean)."""
    import numpy as np
    
    if isinstance(results_dict, dict) and 'X0_pool' not in results_dict:
        rows = []
        for name, res in results_dict.items():
            rows.append(_compute_monte_carlo_stats_latex(res, name))
        df = pd.DataFrame(rows)
    else:
        stats = _compute_monte_carlo_stats_latex(results_dict, controller_name)
        df = pd.DataFrame([stats])
    
    full_path = _output_path(savepath, config, "tables")
    _save_monte_carlo_latex(df, full_path)
    return df

def _save_monte_carlo_latex(df, savepath):
    """Save Monte Carlo table in specific LaTeX format with toprule/midrule/bottomrule."""
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    
    def fmt_stability(x):
        if isinstance(x, (float, np.floating)) and not np.isnan(x):
            return f"{x*100:.1f}\\%"
        return "N/A"
    
    def fmt_float(x, decimals=2):
        if isinstance(x, (float, np.floating)) and not np.isnan(x):
            return f"{x:.{decimals}f}"
        return "N/A"
    
    def escape_tex(s):
        return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
    
    with open(savepath, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        f.write("Model & Stability $S$ & $t_{\\text{conv}}$ (mean)  & Cost $J$ (mean) \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in df.iterrows():
            model = escape_tex(row["Model"])
            stability = fmt_stability(row["Stability $S$"])
            t_conv = fmt_float(row["$t_{\\text{conv}}$ (mean)"], decimals=2)
            cost = fmt_float(row["Cost $J$ (mean)"], decimals=4)
            f.write(f"{model} & {stability}  & {t_conv}  & {cost} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

def _compute_monte_carlo_stats_latex(results_dict, controller_name):
    """Helper for LaTeX formatting - returns only Model, Stability S, t_conv (mean), Cost J (mean)."""
    import numpy as np
    
    NN_final_times = np.asarray(results_dict['NN_final_times'])
    NN_costs = np.asarray(results_dict['NN_costs'])
    
    converged_mask = np.isfinite(NN_final_times)
    n_converged = np.sum(converged_mask)
    n_total = len(NN_final_times)
    stability = n_converged / n_total if n_total > 0 else 0.0
    
    t_conv_mean = np.mean(NN_final_times[converged_mask]) if n_converged > 0 else np.nan
    cost_mean = np.mean(NN_costs[converged_mask]) if n_converged > 0 else np.nan
    
    return {
        "Model": controller_name,
        "Stability $S$": stability,
        "$t_{\\text{conv}}$ (mean)": t_conv_mean,
        "Cost $J$ (mean)": cost_mean
    }