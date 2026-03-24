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
# Shared: single helper for results/{system}/seed_{seed}/{subdir}/filename
# -------------------------------------------------------------------------

def _output_path(savepath, config=None, subdir="plots"):
    """
    If config has system/seed, resolve savepath to results/{system}/seed_{seed}/{subdir}/filename.
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


def _ema_smooth(values, alpha=0.03):
    """Exponential moving average. Returns array of same length."""
    y = np.empty_like(values, dtype=float)
    y[0] = float(values[0])
    for i in range(1, len(values)):
        y[i] = alpha * float(values[i]) + (1.0 - alpha) * y[i - 1]
    return y


def _kwargs_exact_epoch_line(lw: float):
    """
    Matplotlib kwargs for a polyline of *exact* per-epoch losses: straight segments
    between consecutive points, no scatter markers. (EMA/MA are separate: use
    ``smooth='ema'`` only when you want a smoothed trend, not pointwise exact values.)
    """
    return dict(
        linestyle="-",
        drawstyle="default",
        marker=None,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        lw=lw,
    )


def _make_controller_color_map(controller_list):
    """Build name -> color map from default prop_cycle, extended if needed."""
    n = len(controller_list)
    colors_list = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    while len(colors_list) < n:
        colors_list.extend(colors_list)
    return {name: colors_list[k] for k, (name, _) in enumerate(controller_list)}


def _escape_tex(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


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

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
    for k, (name, ctrl) in enumerate(_iter_named_controllers(controllers)):
        V = _value_from_controller_or_gradnet(ctrl, X, device=device, n_quad=n_quad)



        # in plot_value_vs_state_norm:
        ax.scatter(xnorm, V, s=float(s), alpha=float(alpha), label=_controller_name_to_latex(name))

    # hide negative values by cutting y-axis at 0
    _ymin, _ymax = ax.get_ylim()
    ax.set_ylim(0.0, max(0.0, float(_ymax)))

    ax.set_xlabel(r"$\|x\|_w$")
    ax.set_ylabel(r"$V(x)$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.subplots_adjust(left=0.14, right=0.86, bottom=0.16, top=0.96)
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

    # Compensate for z-label on right: add left margin so 3D plot block is centered
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.16, top=0.90, wspace=0.35)

    # --- plot surfaces as before ---
    ax1.plot_surface(Tg, Xig, X_u, cmap=cmap, linewidth=0, antialiased=True)
    ax2.plot_surface(Tg, Xig, X_c, cmap=cmap, linewidth=0, antialiased=True)

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
    """Three stacked plots: value vs norm, value flow, HJB residual. Saves as {savepath}_value_vs_norm.png, _value_flow.png, _hjb_residual.png."""
    if savepath is None and config is not None and hasattr(config, "system"):
        savepath = f"figures/{config.system}_value_analysis{suffix}.pdf"
    from controls.train import loss_unified

    controller_list = list(_iter_named_controllers(controllers))
    color_map = _make_controller_color_map(controller_list)

    def _sub_path(suffix):
        if savepath is None: return None
        p = Path(savepath)
        return str(p.parent / f"{p.stem}_{suffix}.png")

    savepath1 = _sub_path("value_vs_norm")
    savepath2 = _sub_path("value_flow")
    savepath3 = _sub_path("hjb_residual")
    
    # Individual figure size for each plot
    individual_figsize = (6.2, 3.2)
    
    # ============================================================
    # Plot 1: Value vs State Norm
    # ============================================================
    X = sample_conditions(config, n=int(n), seed=int(seed), dist=dist)  # (d,N)
    xnorm = np.asarray(config.ocp.norm(X)).reshape(-1)
    
    fig1, ax1 = plt.subplots(figsize=individual_figsize, constrained_layout=False)
    for name, ctrl in controller_list:
        V = _value_from_controller_or_gradnet(ctrl, X, device=device, n_quad=n_quad)
        color = color_map[name]
        ax1.scatter(xnorm, V, s=float(s), alpha=float(alpha), label=_controller_name_to_latex(name), color=color)
    
    _ymin, _ymax = ax1.get_ylim()
    ax1.set_ylim(0.0, max(0.0, float(_ymax)))
    ax1.set_xlabel(r"$\|x\|_w$")
    ax1.set_ylabel(r"$V(x)$")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.subplots_adjust(left=0.14, right=0.86, bottom=0.16, top=0.96)
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
    
    fig2, ax2 = plt.subplots(figsize=individual_figsize, constrained_layout=False)
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
        ax2.plot(t, V, label=_controller_name_to_latex(name), color=color)
        
        if mark_bumps and V.size >= 2:
            dV = np.diff(V)
            bump_idx = np.where(dV > float(bump_tol))[0] + 1
            if bump_idx.size > 0:
                ax2.plot(t[bump_idx], V[bump_idx], "o", ms=3.5, alpha=0.9, color=color)
    
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$V(x(t))$")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.subplots_adjust(left=0.14, right=0.86, bottom=0.16, top=0.96)
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
    
    fig3, ax3 = plt.subplots(figsize=individual_figsize, constrained_layout=False)
    for name, obj in controller_list:
        color = color_map[name]
        model = _as_grad_model(obj)
        ys = []
        for s in s_grid:
            Xnp = shock[:, None] * float(s)                               # (d,1)
            Xt = torch.tensor(Xnp.T, dtype=torch.float32, device=device)  # (1,d)
            
            total, hjb_err, sup_err, _, _, _ = loss_unified(
                model,
                (Xt,),
                mode="unsupervised",
                supervision=False,
            )
            val = float(hjb_err.detach().cpu().numpy())
            
            if metric == "abs":
                val = np.sqrt(val + float(eps))
            elif metric == "sq":
                pass
            else:
                raise ValueError("metric must be 'abs' or 'sq'")
            
            if log10:
                val = np.log10(val + float(eps))
            
            ys.append(val)
        
        ax3.plot(s_grid, ys, label=_controller_name_to_latex(name), color=color)
    
    ax3.set_xlabel(r"$s$")
    ylab = (r"$\log_{10}(|\mathcal{H}|)$" if (metric == "abs" and log10) else
            r"$\log_{10}(|\mathcal{H}|^2)$" if (metric == "sq" and log10) else
            r"$|\mathcal{H}(s\cdot x)|$" if metric == "abs" else
            r"$|\mathcal{H}(s\cdot x)|^2$")
    ax3.set_ylabel(ylab)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.subplots_adjust(left=0.14, right=0.86, bottom=0.16, top=0.96)
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

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
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
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$x(t,\xi)$")
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.16, top=0.96)

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

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
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
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(lab)
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.16, top=0.96)

    _save_fig(fig, _output_path(savepath, config, "plots"))
    return fig

def plot_training_losses(
    *,
    iters=None,
    losses=None,
    label: str = "loss",
    value_key: str = "loss",
    logy: bool = True,
    color: str | None = None,
    title: str | None = None,
    xlabel: str = r"rollout epoch ($E$)",
    ylabel: str = r"training loss ($\mathcal{L}_{\mathrm{HJB}}$)",
    figsize=(6.2, 3.2),
    savepath=None,
    config=None,
    plot_name: str = "loss_curve",
    series=None,
    smooth: str | None = None,
    ema_alpha: float = 0.03,
    ma_window: int = 200,
    batch_size: int | None = None,
    marker: str | None = None,
    markersize: float = 3.5,
):
    """
    Plot a per-rollout scalar vs rollout index (default: training loss).

    - **Default** ``smooth=None``: **exact** values recorded after each epoch/rollout,
      connected by **straight line segments** (no markers, no EMA). Spikes are preserved.
    - ``smooth='ema'`` / ``smooth='ma'``: **smoothed** curve that does **not** pass
      through each epoch's value—use only if you want a trend line, not exact logs.
    - Pass ``marker='o'`` etc. to overlay explicit point markers.
    - Pass ``value_key='mean_dt'`` (and matching ``ylabel`` / ``logy``) to plot mean
      integrator step size from training history.
    """
    if savepath is None and config is not None and hasattr(config, "system"):
        savepath = f"figures/{config.system}_{plot_name}.pdf"
    # Allow passing history dicts: (name, history_dict) or (name, iters, losses, phase)
    norm_series = []
    for item in series:
        if len(item) == 2 and isinstance(item[1], dict):
            name_k, h = item
            if batch_size is None and "batch_size" in h:
                batch_size = int(h["batch_size"])
            norm_series.append((name_k, h.get("iters"), h.get(value_key), h.get("phase")))
        elif len(item) == 3:
            name_k, it_k, lo_k = item
            norm_series.append((name_k, it_k, lo_k, None))
        elif len(item) == 4:
            name_k, it_k, lo_k, ph_k = item
            norm_series.append((name_k, it_k, lo_k, ph_k))
        else:
            raise ValueError("Each series entry must be (name,iters,losses), (name,history_dict), or (name,iters,losses,phase).")
    if batch_size is None:
        batch_size = 1

    # Extract unsupervised phases and reset iterations - SAME for ALL controllers
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
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
        elif "pre" in str(name_k).lower():
            # For pretrained models without phase info (old checkpoints):
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
        
        # For ALL controllers: reset iterations to start at 1 and scale by batch_size
        if len(it_k) == 0:
            continue
        it_k = (it_k - it_k[0] + 1) * batch_size

        if smooth is None:
            it_plot, y_plot = it_k, lo_k
        elif smooth == "ema":
            it_plot, y_plot = it_k, _ema_smooth(lo_k, alpha=float(ema_alpha))
        elif smooth == "ma":
            w = int(ma_window)
            if w < 2 or w > len(lo_k):
                raise ValueError("ma_window must be in [2, len(losses)]")
            y_plot = np.convolve(lo_k, np.ones(w) / w, mode="valid")
            it_plot = it_k[w - 1 :]
        else:
            raise ValueError("smooth must be None, 'ema', or 'ma'")

        # Exact epoch losses: straight segments, no markers (unless ``marker`` set)
        plot_kw = {**_kwargs_exact_epoch_line(1.8), "label": _controller_name_to_latex(name_k)}
        if marker:
            plot_kw.update(marker=marker, markersize=markersize)
        if color is not None:
            plot_kw["color"] = str(color)
        ax.plot(it_plot, y_plot, **plot_kw)

    if logy:
        ax.set_yscale("log")
    elif value_key == "mean_dt":
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    # Sci y-axis offset text (×10^n): slight headroom so it is not clipped
    top_margin = 0.935 if value_key == "mean_dt" else 0.96
    fig.subplots_adjust(left=0.14, right=0.86, bottom=0.16, top=top_margin)
    _save_fig(fig, _output_path(savepath, config, "plots"))
    return fig, ax


def plot_training_val_mse(
    *,
    config=None,
    histories=None,
    controller_configs=None,
    series=None,
    figsize=(6.2, 3.2),
    savepath=None,
    plot_name: str = "val_mse_curve",
    smooth: str | None = None,
    ema_alpha: float = 0.03,
    batch_size: int | None = None,
    marker: str | None = None,
    markersize: float = 3.5,
):
    """
    Plot validation MSE over rollouts (unsupervised phase only).
    Supervised pretraining phase is stripped using the phase array so that
    pretrained models have the same x-axis length as non-pretrained ones.

    Default ``smooth=None`` plots **exact** val MSE per rollout as straight segments
    (no markers). ``smooth='ema'`` smooths and does not match each rollout exactly.
    Pass ``marker='o'`` to show points at each rollout.
    """
    if series is None and histories is not None and controller_configs is not None:
        series = [(name, histories[name]) for name in controller_configs if controller_configs.get(name, {}).get("enabled") and name in histories and "val_mse" in histories[name]]
    if batch_size is None:
        for _, h in (series or []):
            if isinstance(h, dict) and "batch_size" in h:
                batch_size = int(h["batch_size"])
                break
    if batch_size is None:
        batch_size = 1
    if not series:
        return None, None
    if savepath is None and config is not None and hasattr(config, "system"):
        savepath = f"figures/{config.system}_{plot_name}.pdf"
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
    for name, h in series:
        val_mse = np.asarray(h.get("val_mse"))
        if val_mse.size == 0:
            continue

        phase = h.get("phase")
        if phase is not None:
            ph = np.asarray(phase).reshape(-1)
            n_unsup_iters = int(np.sum(ph != "sup"))
            n_sup_val = max(0, val_mse.size - n_unsup_iters)
            val_mse = val_mse[n_sup_val:]

        if val_mse.size == 0:
            continue
        it = np.arange(1, val_mse.size + 1, dtype=float) * batch_size
        lbl = _controller_name_to_latex(name)
        if smooth == "ema":
            ax.plot(
                it,
                _ema_smooth(val_mse, alpha=float(ema_alpha)),
                **{**_kwargs_exact_epoch_line(1.2), "label": lbl},
            )
        elif marker:
            ax.plot(
                it,
                val_mse,
                marker=marker,
                markersize=markersize,
                linestyle="-",
                lw=1.2,
                label=lbl,
            )
        else:
            ax.plot(it, val_mse, **{**_kwargs_exact_epoch_line(1.2), "label": lbl})
    ax.set_xlabel("rollout epoch ($E$)")
    ax.set_ylabel("validation loss ($\mathcal{L}_{\mathrm{sup}}$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.subplots_adjust(left=0.14, right=0.86, bottom=0.16, top=0.96)
    _save_fig(fig, _output_path(savepath, config, "plots"))
    return fig, ax


# -------------------------------------------------------------------------
# Tables
# -------------------------------------------------------------------------



def save_dataframe_latex(df, savepath, caption=None, label=None, precision=4):
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    disp_df = df.copy()
    
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.2e}" if (abs(x) < 1e-4 or abs(x) > 1e4) else f"{x:.{precision}g}"
        return str(x)

    for col in disp_df.columns:
        if col == "Method":
            continue
        disp_df[col] = disp_df[col].map(fmt).map(_escape_tex)
    disp_df.columns = disp_df.columns.map(_escape_tex)

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

def _fmt_tex_num(x):
    """Format number for LaTeX table: int or |x| >= 1 as plain $n$ or $n.d$; |x| < 1 as $c\\times 10^{e}$."""
    if isinstance(x, int):
        return f"${x}$"
    if isinstance(x, (float, np.floating)):
        if x == 0:
            return "$0$"
        if abs(x) >= 1:
            # Plain number: integer if whole, else short decimal (e.g. t_1 = 15, not 1.5×10^1)
            if abs(x - round(x)) < 1e-9:
                return f"${int(round(x))}$"
            return f"${x:.2g}$"
        # |x| < 1: use power-of-ten (e.g. 1.5e-2 → 1.5×10^{-2})
        exp = int(np.floor(np.log10(abs(x))))
        coeff = x / (10 ** exp)
        if abs(coeff - int(coeff)) < 1e-6:
            return f"${int(coeff)}\\times 10^{{{exp}}}$"
        return f"${coeff:.1f}\\times 10^{{{exp}}}$"
    return str(x)


# Default keys for Jupyter show_spec (includes system and time horizon)
DEFAULT_CONFIG_KEYS = [
    "system", "seed", "n_states", "n_controls", "t1_initial", "t1_max", "nu",
    "control_width", "ic_modes", "ic_scale", "ic_basis",
]


def _config_ic_x0_plain(config) -> str:
    """Plain-text IC family for thesis config row (matches 'sine + cosine' style)."""
    basis = getattr(config, "ic_basis", None)
    if basis == "both":
        return "sine + cosine"
    if basis == "sine":
        return "sine"
    if basis == "cosine":
        return "cosine"
    return "—"


def save_config_table(config, savepath="config.tex", keys=None):
    """
    Save config as a lean thesis tabular.

    Default layout (``keys`` None): seed, n, m, nu, w, K, c, x_0, T_0, T_max.
    If ``keys`` is passed, use the legacy variable-column layout (for custom exports).
    """
    full_path = _output_path(savepath, config, "tables")
    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)

    if keys is not None:
        valid_keys = [k for k in keys if hasattr(config, k)]
        values = [getattr(config, k) for k in valid_keys]
        formatted_values = []
        for k, v in zip(valid_keys, values):
            if k == "system" or k == "ic_basis":
                formatted_values.append(str(v))
            elif k in ["seed", "n_states", "n_controls", "ic_modes"]:
                formatted_values.append(_fmt_tex_num(int(v)))
            elif k in ["t1_initial", "t1_max", "nu", "control_width", "ic_scale"]:
                formatted_values.append(_fmt_tex_num(float(v)))
            else:
                formatted_values.append(_fmt_tex_num(v) if isinstance(v, (int, float, np.floating)) else str(v))
        header_map = {
            "system": "system",
            "seed": "seed",
            "n_states": "$n$",
            "n_controls": "$m$",
            "t1_initial": "$t_0$",
            "t1_max": r"$t_{\max}$",
            "nu": r"$\nu$",
            "control_width": r"$w$",
            "ic_modes": "$M$",
            "ic_scale": "$c$",
            "ic_basis": "IC basis",
            "gamma1": r"$\gamma_1$",
            "gamma2": r"$\gamma_2$",
            "R": "$R$",
        }
        headers = [header_map.get(k, k) for k in valid_keys]
        n_cols = len(valid_keys)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(f"\\begin{{tabular}}{{@{{}}{'c' * n_cols}@{{}}}}\n")
            f.write("\\toprule\n")
            f.write(" & ".join(headers) + " \\\\\n")
            f.write("\\midrule\n")
            f.write(" & ".join(formatted_values) + " \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
        return

    headers = [
        "seed",
        r"$n$",
        r"$m$",
        r"$\nu$",
        r"$w$",
        r"$M$",
        r"$c$",
        r"$x_0$",
        r"$t_0$",
        r"$t_{\max}$",
    ]
    x0_plain = _config_ic_x0_plain(config)
    formatted_values = [
        _fmt_tex_num(int(config.seed)),
        _fmt_tex_num(int(config.n_states)),
        _fmt_tex_num(int(config.n_controls)),
        _fmt_tex_num(float(config.nu)),
        _fmt_tex_num(float(config.control_width)),
        _fmt_tex_num(int(config.ic_modes)),
        _fmt_tex_num(float(config.ic_scale)),
        x0_plain.replace("&", r"\&"),
        _fmt_tex_num(float(config.t1_initial)),
        _fmt_tex_num(float(config.t1_max)),
    ]
    n_cols = len(headers)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(f"\\begin{{tabular}}{{@{{}}{'c' * n_cols}@{{}}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        f.write(" & ".join(formatted_values) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

def _controller_name_to_latex(name):
    """Map controller display name to LaTeX (tables and plots): LQR → $u_{\\mathrm{LQR}}$, GradQRNet (+ suffix) → $u_{\\theta}$ (+ suffix). No \\bm (mathtext-safe)."""
    name = str(name).strip()
    if not name:
        return ""
    if name == "LQR" or name.startswith("LQR "):
        return r"$u_{\mathrm{LQR}}$"
    if name == "GradQRNet":
        return r"$u_{\theta}$"
    if name.startswith("GradQRNet"):
        suffix = name[len("GradQRNet"):].strip()
        return r"$u_{\theta}$" + (f" {suffix}" if suffix else "")
    return name


def save_results_table(results_list, config=None, savepath="results.tex"):
    """
    Reformats the flat results list into a professional Thesis table.
    Groups by Controller and stacks Standard/Hard datasets as rows.
    """
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
    if "Method" in df.columns:
        df["Method"] = df["Method"].apply(_controller_name_to_latex)

    cols = ["Method", "Data", "$S$", "$J_{\\text{avg}}$"]
    df = df[[c for c in cols if c in df.columns]]

    # 3. Format 'S' as LaTeX math percentage, e.g. $98\%$
    if "$S$" in df.columns:
        df["$S$"] = df["$S$"].apply(
            lambda x: f"${float(x)*100:.0f}\\%$" if isinstance(x, (float, int)) else x
        )

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
    
    X_all = np.asarray(data["X"])
    t_all = np.asarray(data["t"]).reshape(-1)
    X_norms = config.norm(X_all).reshape(-1)
    
    n_traj = int(data.get("n_trajectories", -1))
    n_points = int(X_all.shape[1])
    t_max = t_all.max()
    abs_x_mean = np.mean(np.abs(X_all))
    abs_x_max = np.max(np.abs(X_all))
    norm_x_mean = np.mean(X_norms)
    norm_x_max = np.max(X_norms)
    
    full_path = _output_path(savepath, config, "tables")
    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
    
    with open(full_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{@{}ccccccc@{}}\n")
        f.write("\\toprule\n")
        f.write("$N_{\\mathrm{traj}}$ & $|\\mathcal{D}|$ & $t$ (max) & $|x|$ (mean) & $|x|$ (max) & $\\|x\\|$ (mean) & $\\|x\\|$ (max)\\\\\n")
        f.write("\\midrule\n")
        n_tr = _fmt_tex_num(int(n_traj))
        n_pt = _fmt_tex_num(int(n_points))
        t_mx = _fmt_tex_num(float(t_max))
        ax_mn = _fmt_tex_num(float(abs_x_mean))
        ax_mx = _fmt_tex_num(float(abs_x_max))
        nx_mn = _fmt_tex_num(float(norm_x_mean))
        nx_mx = _fmt_tex_num(float(norm_x_max))
        f.write(f"{n_tr} & {n_pt} & {t_mx} & {ax_mn} & {ax_mx} & {nx_mn} & {nx_mx} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    
    # Return a DataFrame for compatibility
    return pd.DataFrame({
        "$N_{\\mathrm{traj}}$": [n_traj],
        "$|\\mathcal{D}|$": [n_points],
        "t (max)": [t_max],
        "|x| (mean)": [abs_x_mean],
        "|x| (max)": [abs_x_max],
        "$\\|x\\|$ (mean)": [norm_x_mean],
        "$\\|x\\|$ (max)": [norm_x_max]
    })

def save_params_table(obj, savepath, title="Configuration", keys=None, config=None, train_n_cand_tex=None):
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
        _save_train_config_latex(obj, full_path, n_cand_tex=train_n_cand_tex)
    else:
        save_dataframe_latex(df, full_path, caption=title, label=f"tab:{Path(savepath).stem}")


def save_table_latex(df, savepath, config=None, caption=None, label=None, **to_latex_kw):
    """Save a DataFrame to LaTeX in the config results tables dir (if config given). Kwargs passed to df.to_latex()."""
    full_path = _output_path(savepath, config, "tables")
    Path(full_path).parent.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    # Apply LaTeX controller shorthands to Method/Controller column if present
    for col in ("Method", "Controller"):
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(_controller_name_to_latex)
            break
    # Escape % for LaTeX in all cells (to_latex(escape=False) would otherwise write literal %)
    for col in df_out.columns:
        df_out[col] = df_out[col].apply(lambda x: str(x).replace("%", "\\%"))
    kwargs = {"index": False, "escape": False, "caption": caption, "label": label, **to_latex_kw}
    df_out.to_latex(full_path, **kwargs)
    return df


def save_robustness_table_latex(df, savepath, config=None, caption=None, label=None):
    """
    Save the robustness DataFrame (MultiIndex columns: Stability / Cost $J$ × c values)
    as a lean LaTeX tabular only (no \\begin{table}). Caller adds table, caption, label.
    Layout: one block of subcolumns for Stability (per c), one for Cost (per c).
    """
    full_path = _output_path(savepath, config, "tables")
    Path(full_path).parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("save_robustness_table_latex expects MultiIndex columns (metric, c)")
    level1 = df.columns.get_level_values(1)
    n_c = len(level1) // 2
    c_values = level1[:n_c].tolist()
    level0 = df.columns.get_level_values(0)
    stab_name = level0[0]
    cost_name = level0[n_c]

    with open(full_path, "w", encoding="utf-8") as f:
        col_spec = "l|" + "c" * n_c + "|" + "c" * n_c
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")
        f.write(f"Controller & \\multicolumn{{{n_c}}}{{c}}{{{stab_name}}} & \\multicolumn{{{n_c}}}{{c}}{{{cost_name}}} \\\\\n")
        f.write("\\cmidrule(lr){2-" + str(1 + n_c) + "} \\cmidrule(lr){" + str(2 + n_c) + "-" + str(1 + 2 * n_c) + "}\n")
        subheaders = " & ".join([f"$c={c}$" for c in c_values]) + " & " + " & ".join([f"$c={c}$" for c in c_values])
        f.write(f" & {subheaders} \\\\\n")
        f.write("\\midrule\n")
        for idx in df.index:
            f.write(_controller_name_to_latex(idx))
            for col in df.columns:
                val = df.loc[idx, col]
                if str(val) == "N/A" or str(val) == "---":
                    f.write(" & ---")
                elif isinstance(val, str) and "%" in val:
                    # Stability etc.: output in math mode as $98\%$
                    num_str = val.replace("%", "").replace("\\", "").strip()
                    f.write(f" & ${num_str}\\%$")
                else:
                    f.write(" & " + str(val).replace("%", "\\%"))
            f.write(" \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    return df


def _save_train_config_latex(cfg, savepath, n_cand_tex=None):
    """
    Thesis tabular: $E$, $h$, $\\lambda_{\\text{sup}}$, $\\mu$, $|\\mathcal{B}|$,
    $N_{\\mathrm{cand}}$, $\\Delta t_{\\min}$, $\\Delta t_{\\max}$ ($E$ = unsupervised rollout count).

    ``TrainConfig`` always defines ``dt_min`` / ``dt_max`` (HJB rollout integrator clamps).

    Pass ``n_cand_tex`` to override the $N_{\\mathrm{cand}}$ cell (e.g. ``r'$\\{1,8,\\ldots\\}$'``).
    """
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
    
    epochs_unsup = getattr(cfg, 'rollouts', 5)
    horizon = getattr(cfg, 'horizon', 30)
    lr = getattr(cfg, 'unsup_lr', 5e-4)
    lambda_sup = getattr(cfg, 'lambda_sup_base', 0.5)
    batch_size = getattr(cfg, 'batch_size', None)
    n_cand = getattr(cfg, 'n_candidates', getattr(cfg, 'n_cand', None))
    dt_min = float(getattr(cfg, "dt_min", 1e-2))
    dt_max = float(getattr(cfg, "dt_max", 1.0))
    batch_size_tex = _fmt_tex_num(int(batch_size)) if batch_size is not None else "---"
    if n_cand_tex is None:
        n_cand_tex = _fmt_tex_num(int(n_cand)) if n_cand is not None else "---"
    
    with open(savepath, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{@{}cccccccc@{}}\n")
        f.write("\\toprule\n")
        f.write(
            "$E$ & $h$ & $\\lambda_{\\text{sup}}$ & $\\mu$ & $|\\mathcal{B}|$ & $N_{\\mathrm{cand}}$ & "
            "$\\Delta t_{\\min}$ & $\\Delta t_{\\max}$ \\\\\n"
        )
        f.write("\\midrule\n")
        f.write(
            f"{_fmt_tex_num(int(epochs_unsup))} & {_fmt_tex_num(int(horizon))} & {_fmt_tex_num(float(lambda_sup))} & {_fmt_tex_num(float(lr))} & {batch_size_tex} & {n_cand_tex} & "
            f"{_fmt_tex_num(dt_min)} & {_fmt_tex_num(dt_max)} \\\\\n"
        )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def show_monte_carlo_results(results_dict, controller_name="Controller", title=None):
    """
    Display Monte Carlo results: controllers as columns, metrics (S, J) as rows.

    Returns ``None`` so Jupyter does not render a second (raw) DataFrame after the styled table.

    Parameters
    ----------
    results_dict : dict or dict of dicts
        Output from simulation.monte_carlo() - can be single controller or multiple
    controller_name : str, optional
        Name if single controller (ignored if results_dict contains multiple)
    title : str, optional
        Title for the table
    """
    if isinstance(results_dict, dict) and 'X0_pool' not in results_dict:
        stats = {name: _compute_monte_carlo_stats(res, name) for name, res in results_dict.items()}
    else:
        stats = {controller_name: _compute_monte_carlo_stats(results_dict, controller_name)}

    df = pd.DataFrame({
        name: {"Stability (S)": s["Stability (S)"], "Cost J (mean)": s["Cost J (mean)"]}
        for name, s in stats.items()
    })

    if title:
        display(Markdown(f"**{title}**"))

    display(df.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f4f4f4'), ('color', 'black'), ('font-weight', 'bold')]}
    ]))


def compute_monte_carlo_errors(results_dict):
    """
    Standard errors for one controller's ``monte_carlo`` output.

    Returns
    -------
    S : float
        Stability rate in ``[0, 1]``.
    se_S : float
        ``sqrt(S * (1-S) / n_MC)`` (Bernoulli SE on S).
    cost_mean : float
        Mean cost over successful trajectories (``nan`` if none).
    se_J : float
        Sample SE of cost: ``std(costs, ddof=1) / sqrt(n_success)`` (``nan`` if ``n_success < 2``).
    n_mc : int
    n_success : int
    """
    return _monte_carlo_stats_core(results_dict)


def _monte_carlo_stats_core(results_dict):
    """
    Stability S = n_success/n_MC with SE = sqrt(S(1-S)/n_MC).
    Cost mean and SE = sample std of successful costs / sqrt(n_success) (ddof=1).
    Returns (S, se_S, cost_mean, se_J, n_mc, n_success).
    """
    NN_final_times = np.asarray(results_dict["NN_final_times"])
    NN_costs = np.asarray(results_dict["NN_costs"])
    converged_mask = np.isfinite(NN_final_times)
    n_success = int(np.sum(converged_mask))
    n_mc = int(NN_final_times.size)
    S = n_success / n_mc if n_mc > 0 else 0.0
    se_S = float(np.sqrt(S * (1.0 - S) / n_mc)) if n_mc > 0 else 0.0
    costs_ok = NN_costs[converged_mask]
    if n_success == 0:
        cost_mean = np.nan
        se_J = np.nan
    else:
        cost_mean = float(np.mean(costs_ok))
        if n_success < 2:
            se_J = np.nan
        else:
            se_J = float(np.std(costs_ok, ddof=1) / np.sqrt(n_success))
    return S, se_S, cost_mean, se_J, n_mc, n_success


def _raw_monte_carlo_stats(results_dict):
    """Backward-compatible: (stability, cost_mean) only."""
    S, _, cost_mean, _, _, _ = _monte_carlo_stats_core(results_dict)
    return S, cost_mean


def _format_stability_pm_html(S, se_S):
    if not np.isfinite(S):
        return "N/A"
    return f"{100 * S:.0f} ± {100 * se_S:.0f}%"


def _format_cost_pm_html(cost_mean, se_J):
    if not np.isfinite(cost_mean):
        return "N/A"
    if not np.isfinite(se_J) or se_J <= 0:
        return f"{cost_mean:.2f}"
    return f"{cost_mean:.2f} ± {se_J:.2f}"


def _format_stability_pm_latex(S, se_S):
    if not np.isfinite(S):
        return "N/A"
    return f"${100 * S:.0f} \\pm {100 * se_S:.0f}\\%$"


def _format_cost_pm_latex(cost_mean, se_J):
    if not np.isfinite(cost_mean):
        return "N/A"
    if not np.isfinite(se_J) or se_J <= 0:
        return f"${cost_mean:.2f}$"
    return f"${cost_mean:.2f} \\pm {se_J:.2f}$"


def _compute_monte_carlo_stats(results_dict, controller_name):
    """Notebook table: stability and cost with standard errors."""
    S, se_S, cost_mean, se_J, _, _ = _monte_carlo_stats_core(results_dict)
    return {
        "Controller": controller_name,
        "Stability (S)": _format_stability_pm_html(S, se_S),
        "Cost J (mean)": _format_cost_pm_html(cost_mean, se_J),
    }


def save_monte_carlo_results(results_dict, controller_name="Controller", config=None, 
                            savepath="monte_carlo.tex"):
    """Save Monte Carlo results to LaTeX table: controllers as columns, S and J as rows.

    Returns ``None`` so a following ``show_monte_carlo_results`` call does not leave a duplicate
    DataFrame as the cell's ``Out[n]`` when this runs last.
    """
    
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

def _save_monte_carlo_latex(df, savepath):
    """Save Monte Carlo table: controllers as columns, metrics (S, J) as rows with ± SE."""
    os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)

    models = [_controller_name_to_latex(m) for m in df["Model"]]
    n = len(models)

    with open(savepath, "w", encoding="utf-8") as f:
        f.write(f"\\begin{{tabular}}{{l{'r' * n}}}\n")
        f.write("\\toprule\n")
        f.write(" & " + " & ".join(models) + " \\\\\n")
        f.write("\\midrule\n")
        stab_vals = " & ".join(str(v) for v in df["Stability"])
        f.write(f"Stability & {stab_vals} \\\\\n")
        cost_vals = " & ".join(str(v) for v in df["Cost $J$"])
        f.write(f"Cost $J$ & {cost_vals} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

def _compute_monte_carlo_stats_latex(results_dict, controller_name):
    """LaTeX table cells: pre-formatted $value \\pm SE$ strings."""
    S, se_S, cost_mean, se_J, _, _ = _monte_carlo_stats_core(results_dict)
    return {
        "Model": controller_name,
        "Stability": _format_stability_pm_latex(S, se_S),
        "Cost $J$": _format_cost_pm_latex(cost_mean, se_J),
    }