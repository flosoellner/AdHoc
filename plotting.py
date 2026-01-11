from __future__ import annotations
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from simulation import sim_closed_loop
from sampling import sample_conditions



from pathlib import Path

def _save_fig(fig, savepath, *, dpi=300):
    if savepath is None:
        return
    p = Path(savepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=dpi)



# put near the top of each function (or make it a module constant)
colors = ["#56B4E9", "#7B2CBF"]  # light blue, purple

from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list(
    "blue_purple_blend",
    ["#56B4E9", "#7B2CBF"],   # light blue -> purple
    N=256
)

def set_thesis_style(usetex=True, font_size=11):
    mpl.rcParams.update({
        "text.usetex": bool(usetex),
        "font.family": "serif",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": font_size - 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "lines.linewidth": 1.2,
    })

# plotting.py


import torch

import numpy as np
import torch
import matplotlib.pyplot as plt

from sampling import sample_conditions

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


def plot_value_slice(
    *,
    config,
    controllers,                 # list like [("LQR", config.ocp.LQR), ("ctrl1", ctrl1)]
    i=None,                      # basis slice x = s e_i
    v=None,                      # direction slice x = s v (overrides i)
    smax=12.0,
    ns=401,
    n_quad=64,
    device="cpu",
    title=None,
    figsize=(6.2, 3.2),
    savepath=None,
):
    """
    Plots s -> V(x(s)) for multiple controllers on a 1D slice through R^d.
    Slice options:
      - basis slice: set i (int): x = s e_i
      - direction slice: set v (d,) : x = s v
    """
    d = int(config.n_states)
    s = np.linspace(-float(smax), float(smax), int(ns))

    if v is not None:
        v = np.asarray(v).reshape(-1)
        if v.size != d:
            raise ValueError(f"v must have shape ({d},)")
        v = v / (np.linalg.norm(v) + 1e-12)
        X = v[:, None] * s[None, :]          # (d,ns)
        xlabel = r"$s$ in $x=s\,v$"
    else:
        if i is None:
            i = d // 2
        i = int(i)
        if not (0 <= i < d):
            raise ValueError(f"i must be in [0,{d-1}]")
        X = np.zeros((d, s.size))
        X[i, :] = s
        xlabel = rf"$x_{{{i}}}$ (slice; other components $0$)"

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
# then replace the plotting loops with this
    for k, (name, ctrl) in enumerate(controllers):
        c = colors[k % len(colors)]
        V = _value_from_controller_or_gradnet(ctrl, X, device=device, n_quad=n_quad)

        # in plot_value_slice:
        ax.plot(s, V, label=str(name), color=c)



    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$V(x)$")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save_fig(fig, savepath)
    return fig


def plot_value_vs_state_norm(
    *,
    config,
    controllers,                 # list like [("LQR", config.ocp.LQR), ("ctrl1", ctrl1)]
    n=8000,
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
    # then replace the plotting loops with this
    for k, (name, ctrl) in enumerate(controllers):
        c = colors[k % len(colors)]
        V = _value_from_controller_or_gradnet(ctrl, X, device=device, n_quad=n_quad)



        # in plot_value_vs_state_norm:
        ax.scatter(xnorm, V, s=float(s), alpha=float(alpha), label=str(name), color=c)

    ax.set_xlabel(r"$\|x\|_w$")
    ax.set_ylabel(r"$V(x)$")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save_fig(fig, savepath)
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
    tspan=(0.0, 30.0),
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
    # 1) sample one IC
    X0 = sample_conditions(config, n=1, seed=seed, dist=dist)[:, 0]

    # 2) common time grid
    t_eval = np.linspace(float(tspan[0]), float(tspan[1]), int(Nt))

    # 3) events
    if events == "auto":
        events = config.ocp.make_integration_events()

    # 4) simulate
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

    # 5) plot
    xi = np.asarray(config.xi).reshape(-1)
    if X_u.shape[0] != xi.size:
        raise ValueError(f"Mismatch: X has {X_u.shape[0]} states but xi has {xi.size} points")

    Tg, Xig = np.meshgrid(t_eval, xi)

    from matplotlib.ticker import MaxNLocator

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
    ax1.set_title("Uncontrolled", pad=-6)
    ax2.set_title("Controlled", pad=-6)

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

    print("saving to:", savepath)  # temporary debug
    _save_fig(fig, savepath)       # ALWAYS call this if savepath is not None

    return fig




# -------------------------
# Small helpers
# -------------------------
def _as_2d_state(X) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return X

def _weighted_norm_w(config, X: np.ndarray) -> np.ndarray:
    # X: (d, N)
    w = getattr(getattr(config, "ocp", None), "w_flat", None)
    if w is None:
        return np.linalg.norm(X, axis=0)
    w = np.asarray(w).reshape(-1)
    return np.sqrt(np.sum((w[:, None]) * (X ** 2), axis=0) + 1e-12)

def _eval_u(controller, X: np.ndarray) -> np.ndarray:
    # expects controller.eval_U(X) with X: (d,N) -> U: (m,N) or (m,)
    U = np.asarray(controller.eval_U(X))
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    return U

def _eval_grad(controller, X: np.ndarray) -> np.ndarray:
    # returns dVdX: (d,N)
    if hasattr(controller, "eval_dVdX"):
        G = np.asarray(controller.eval_dVdX(X))
        return G
    if hasattr(controller, "grad_net") and (controller.grad_net is not None):
        import torch
        x_t = torch.tensor(X.T, dtype=torch.float32)
        with torch.no_grad():
            g = controller.grad_net(x_t).cpu().numpy().T
        return g
    raise ValueError("Controller must provide eval_dVdX(X) or have .grad_net")

def _eval_value(controller, X: np.ndarray) -> np.ndarray | None:
    # returns V: (N,)
    if hasattr(controller, "eval_V"):
        V = np.asarray(controller.eval_V(X)).reshape(-1)
        return V
    if hasattr(controller, "value_net") and (controller.value_net is not None):
        import torch
        x_t = torch.tensor(X.T, dtype=torch.float32)
        with torch.no_grad():
            v = controller.value_net(x_t).cpu().numpy().reshape(-1)
        return v
    return None

def _bin2d_mean(x, y, v, xedges, yedges):
    # x,y,v: (N,)
    xi = np.digitize(x, xedges) - 1
    yi = np.digitize(y, yedges) - 1
    ok = (xi >= 0) & (xi < len(xedges) - 1) & (yi >= 0) & (yi < len(yedges) - 1) & np.isfinite(v)
    xi = xi[ok]; yi = yi[ok]; v = v[ok]

    S = np.zeros((len(yedges) - 1, len(xedges) - 1), dtype=float)
    C = np.zeros_like(S)
    np.add.at(S, (yi, xi), v)
    np.add.at(C, (yi, xi), 1.0)
    Z = S / np.maximum(C, 1.0)
    Z[C == 0] = np.nan
    return Z


# -------------------------
# 1) Tradeoff scatter plots
# -------------------------
def plot_tradeoff_scatter(
    df,
    *,
    x: str,
    y: str,
    group: str = "architecture",
    label_map: dict | None = None,
    markers: dict | None = None,
    colors: dict | None = None,
    lqr_selector=None,          # callable(row)->bool or boolean mask
    stable_only: bool = False,
    stable_col: str = "max_final_dist",
    stable_thr: float | None = None,
    xscale: str = "log",
    xlim=None,
    ylim=None,
    yline: float | None = None,
    yline_lqr: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize=(6.0, 2.6),
):
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    d = df.copy()
    d = d[np.isfinite(d[x]) & np.isfinite(d[y])]

    if stable_only and (stable_thr is not None) and (stable_col in d.columns):
        d = d[d[stable_col] < stable_thr]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xscale(xscale)

    # LQR baseline
    if lqr_selector is not None:
        lqr_rows = d[lqr_selector(d) if callable(lqr_selector) else lqr_selector]
        if len(lqr_rows) > 0:
            xl = float(lqr_rows[x].iloc[0])
            yl = float(lqr_rows[y].iloc[0])
            ax.plot([xl], [yl], "ko", ms=7, mfc="k", label="LQR")
            if yline_lqr:
                ax.axhline(yl, ls="--", c="k", lw=1.2)

    if yline is not None:
        ax.axhline(float(yline), c="k", lw=1.2)

    groups = list(d[group].dropna().unique())
    for g in groups:
        if lqr_selector is not None and ((callable(lqr_selector) and lqr_selector(d[d[group] == g]).all()) is True):
            continue
        dd = d[d[group] == g]
        lab = label_map.get(g, g) if label_map else str(g)
        mk = markers.get(g, "o") if markers else "o"
        col = colors.get(g, None) if colors else None
        ax.plot(dd[x].to_numpy(), dd[y].to_numpy(), mk, ms=6, lw=0, label=lab, color=col)

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if title: ax.set_title(title)

    ax.legend(loc="best", frameon=True)
    return fig


# -------------------------
# 2) Grouped bars with IQR + min/max
# -------------------------
def plot_grouped_bars_iqr(
    df,
    *,
    x: str,             # group on x (e.g. n_trajectories_train)
    hue: str,           # architecture
    y: str,             # metric
    order=None,
    hue_order=None,
    ylabel: str | None = None,
    title: str | None = None,
    yscale: str | None = None,
    yline: float | None = None,
    yline_lqr: float | None = None,
    figsize=(6.0, 3.0),
):
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    d = df.copy()
    d = d[np.isfinite(d[y])]

    if order is None:
        order = sorted(d[x].dropna().unique())
    if hue_order is None:
        hue_order = list(d[hue].dropna().unique())

    # compute stats
    med = np.full((len(order), len(hue_order)), np.nan)
    p25 = np.full_like(med, np.nan)
    p75 = np.full_like(med, np.nan)
    vmin = np.full_like(med, np.nan)
    vmax = np.full_like(med, np.nan)

    for j, h in enumerate(hue_order):
        dh = d[d[hue] == h]
        for i, xv in enumerate(order):
            dv = dh[dh[x] == xv][y].to_numpy()
            if dv.size:
                med[i, j] = np.nanmedian(dv)
                p25[i, j] = np.nanpercentile(dv, 25)
                p75[i, j] = np.nanpercentile(dv, 75)
                vmin[i, j] = np.nanmin(dv)
                vmax[i, j] = np.nanmax(dv)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.grid(True, axis="y", alpha=0.3)

    ng = len(order)
    nb = len(hue_order)
    xbase = np.arange(ng)
    width = min(0.8, nb / (nb + 1.5))
    barw = width / max(nb, 1)

    bars = []
    for j, h in enumerate(hue_order):
        xj = xbase - width / 2 + (j + 0.5) * barw
        b = ax.bar(xj, med[:, j], width=barw * 0.95, label=str(h))
        bars.append(b)

        # IQR errorbars
        yerr_lo = med[:, j] - p25[:, j]
        yerr_hi = p75[:, j] - med[:, j]
        ax.errorbar(xj, med[:, j], yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="k", lw=1.0, capsize=6)

        # min/max markers
        ax.scatter(xj, vmax[:, j], c="k", marker="^", s=18, zorder=3)
        ax.scatter(xj, vmin[:, j], c="k", marker="v", s=18, zorder=3)

    ax.set_xticks(xbase)
    ax.set_xticklabels([str(v) for v in order])
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    if title: ax.set_title(title)
    if yscale: ax.set_yscale(yscale)
    if yline is not None: ax.axhline(float(yline), c="k", lw=1.2)
    if yline_lqr is not None: ax.axhline(float(yline_lqr), c="k", lw=1.2, ls="--")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=min(6, max(1, nb)), frameon=True)
    return fig


# -------------------------
# 3) Basin / robustness plot in 2D summary space
# -------------------------
def plot_basin_scatter(
    *,
    config,
    X0_pool: np.ndarray,        # (d,N)
    success: np.ndarray,        # (N,) bool
    title: str | None = None,
    figsize=(5.6, 3.6),
):
    X0 = _as_2d_state(X0_pool)
    ok = np.asarray(success).astype(bool).reshape(-1)
    if X0.shape[1] != ok.size:
        raise ValueError("X0_pool must be (d,N) and success must be (N,)")

    z1 = _weighted_norm_w(config, X0)
    z2 = np.max(np.abs(X0), axis=0)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.grid(True, alpha=0.3)
    ax.scatter(z1[~ok], z2[~ok], s=10, c="tab:red", label="fail", alpha=0.6)
    ax.scatter(z1[ok],  z2[ok],  s=10, c="tab:blue", label="success", alpha=0.6)
    ax.set_xlabel(r"$\|x_0\|_{w}$")
    ax.set_ylabel(r"$\max_i |x_{0,i}|$")
    if title: ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    return fig



def plot_hjb_residual_shock_line(
    *,
    config,
    controllers,                 # [("LQR", config.ocp.LQR), ("ctrl1", ctrl1.grad_net)]
    shock=None,                  # (d,) numpy; if None, sample one and scale it
    shock_seed: int = 0,
    s_grid=None,                 # array of intensities; default below
    dist_for_shock: float = 1.0, # used only if shock=None
    device: str = "cpu",
    metric: str = "abs",         # "abs" -> |H|, "sq" -> H^2
    log10: bool = False,
    eps: float = 1e-12,
    colors=("#56B4E9", "#7B2CBF"),  # light blue, purple
    title: str | None = None,
    figsize=(6.2, 3.2),
    savepath=None,
):
    """
    1D "truth plot": shock intensity s vs HJB residual error at x = s * shock.

    IMPORTANT: This calls the preexisting `loss_hjb_residual` from `controls/train.py`,
    so the residual definition (BurgersPhysics, u clamp, etc.) matches training exactly.

    Notes:
      - `loss_hjb_residual(model, (X,))` returns mean(H^2) over the batch.
      - Here we use batch size 1, so it returns H(x)^2.
      - If metric="abs", we plot |H(x)| = sqrt(H(x)^2).
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from sampling import sample_conditions
    from controls.train import loss_unified

    if s_grid is None:
        s_grid = np.linspace(0.0, 4.0, 20)
    s_grid = np.asarray(s_grid, dtype=float).reshape(-1)

    d = int(config.n_states)

    # fixed shock shape
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
        # If it's an LQR controller, wrap as model; if it's a GradNet, use as-is.
        if hasattr(obj, "eval_dVdX"):   # LQR
            return _LQRAsGradModel(config)
        # assume it's a torch model like GradNet with .config
        if not hasattr(obj, "config"):
            raise ValueError("For non-LQR, pass the GradNet itself (e.g. ctrl1.grad_net), not Control().")
        return obj

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for k, (name, obj) in enumerate(controllers):
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

            # for the "truth line" you want H(x)^2, which is hjb_err
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

        c = colors[k % len(colors)]
        ax.plot(s_grid, ys, label=str(name), color=c)

    ax.set_xlabel(r"shock intensity $s$ (state $x = s\,x_{\mathrm{shock}}$)")
    ylab = (r"$\log_{10}(|H|)$" if (metric == "abs" and log10) else
            r"$\log_{10}(H^2)$" if (metric == "sq" and log10) else
            r"$|H(x)|$" if metric == "abs" else
            r"$H(x)^2$")
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)

    _save_fig(fig, savepath)
    return fig


def plot_value_flow(
    *,
    config,
    controllers,                  # [("LQR", config.ocp.LQR), ("ctrl1", ctrl1)]  (each must have eval_U)
    x0=None,                      # (d,) or (d,1) numpy; if None, sampled
    seed: int = 0,
    dist=None,
    tspan=(0.0, 30.0),
    Nt: int = 400,
    solver="LSODA",
    events="auto",
    device="cpu",
    n_quad: int = 64,             # only used if V reconstructed from grad_net
    bump_tol: float = 0.0,        # mark bumps where V(t_k)-V(t_{k-1}) > bump_tol
    mark_bumps: bool = True,
    colors=("#56B4E9", "#7B2CBF"),  # light blue, purple
    title: str | None = None,
    figsize=(6.4, 3.2),
    savepath=None,
):
    """
    Value flow plot: time t vs V(x(t)) along a CLOSED-LOOP trajectory.

    - Simulates x(t) with sim_closed_loop using controller.eval_U.
    - Evaluates V(x(t)) using:
        * controller.eval_V(X) if available (e.g. LQR), else
        * reconstructs V from controller.grad_net via V_from_gradnet_line_integral
          (through _value_from_controller_or_gradnet).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sampling import sample_conditions

    # --- initial condition ---
    if x0 is None:
        X0 = sample_conditions(config, n=1, seed=int(seed), dist=dist)  # (d,1)
        x0 = X0[:, 0]
    x0 = np.asarray(x0).reshape(-1)

    # --- time grid ---
    t_eval = np.linspace(float(tspan[0]), float(tspan[1]), int(Nt))

    # --- events ---
    if events == "auto":
        if hasattr(config, "ocp") and hasattr(config.ocp, "make_integration_events"):
            events = config.ocp.make_integration_events()
        else:
            events = None

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for k, (name, ctrl) in enumerate(controllers):
        # simulate closed loop
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

        # evaluate V along trajectory (Nt_eff,)
        V = _value_from_controller_or_gradnet(ctrl, X, device=device, n_quad=int(n_quad))

        c = colors[k % len(colors)]
        ax.plot(t, V, label=str(name), color=c)

        if mark_bumps and V.size >= 2:
            dV = np.diff(V)
            bump_idx = np.where(dV > float(bump_tol))[0] + 1
            if bump_idx.size > 0:
                ax.plot(t[bump_idx], V[bump_idx], "o", ms=3.5, color=c, alpha=0.9)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$V(x(t))$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)

    _save_fig(fig, savepath)
    return fig

def plot_value_gradient_profile(
    *,
    config,
    controllers,                 # [("LQR", config.ocp.LQR), ("ctrl3", ctrl3)]  (must provide grad via eval_dVdX or .grad_net)
    which: str = "hard",         # "hard" | "random"
    hard_index: int = 0,
    n_pool: int = 10000,
    dist=None,
    seed: int = 0,
    eps: float = 1e-2,
    cache_dir: str = "./cache_hard_ics",
    device: str = "cpu",
    colors=("#56B4E9", "#7B2CBF"),
    title: str | None = None,
    figsize=(6.4, 3.2),
    savepath=None,
):
    """
    Plot gradient profile dV/dx over spatial coordinate xi for one IC x0.

    - If which="hard": uses evaluation.ic_hard(...) (LQR-failing ICs), picks column hard_index.
    - If which="random": samples one IC via sample_conditions.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sampling import sample_conditions

    if which not in ("hard", "random"):
        raise ValueError("which must be 'hard' or 'random'")

    # choose x0 (shape (d,))
    if which == "hard":
        from evaluation import ic_hard
        X_hard, _meta = ic_hard(config, n_pool=int(n_pool), eps=float(eps), dist=dist, seed=int(seed), cache_dir=cache_dir)
        if X_hard.shape[1] == 0:
            raise ValueError("ic_hard found 0 hard ICs (try increasing n_pool or changing dist/seed).")
        j = int(hard_index) % int(X_hard.shape[1])
        x0 = X_hard[:, j]
        subtitle = rf"hard IC #{j} (from LQR failures)"
    else:
        X0 = sample_conditions(config, n=1, seed=int(seed), dist=dist)  # (d,1)
        x0 = X0[:, 0]
        subtitle = rf"random IC (seed={seed})"

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    d = int(config.n_states)
    if x0.size != d:
        raise ValueError(f"x0 must have shape ({d},)")

    xi = np.asarray(config.xi).reshape(-1)
    if xi.size != d:
        # burgers xi should match n_states
        raise ValueError(f"Mismatch: xi has {xi.size} points but state has {d} entries")

    # evaluate gradients (d,) for each controller
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for k, (name, ctrl) in enumerate(controllers):
        # use existing helper in plotting.py (supports LQR.eval_dVdX or ctrl.grad_net)
        G = _eval_grad(ctrl, x0.reshape(-1, 1))  # (d,1)
        g = np.asarray(G).reshape(d)

        c = colors[k % len(colors)]
        ax.plot(xi, g, label=str(name), color=c)

    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\partial V / \partial x(\xi)$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title or ("Value gradient profile — " + subtitle))

    _save_fig(fig, savepath)
    return fig

class _ZeroController:
    def __init__(self, config):
        self.m = int(config.n_controls)
    def eval_U(self, X):
        import numpy as np
        X = np.asarray(X)
        if X.ndim == 1:
            return np.zeros((self.m,), dtype=float)
        return np.zeros((self.m, X.shape[1]), dtype=float)



def plot_state_profile_snapshots(
    *,
    config,
    controllers,                  # [("LQR", config.ocp.LQR), ("ctrl3", ctrl3)]
    times=(0.0, 2.0, 10.0),        # seconds
    x0=None,                      # (d,) or (d,1); if None, sampled
    seed: int = 0,
    dist=None,
    solver="LSODA",
    events="auto",
    colors=("#56B4E9", "#7B2CBF"), # controller colors (light blue, purple)
    time_cmap="Greys",            # snapshots within one controller (light->dark)
    lw: float = 1.8,
    title: str | None = None,
    figsize=(6.6, 3.6),
    savepath=None,
):
    """
    1D "state profile" snapshots: x(t,xi) vs xi at a few specified times.

    For each controller, simulates one trajectory and plots x(t_k, xi) for each t_k in `times`.
    Within each controller, later times are darker (using `time_cmap`).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sampling import sample_conditions

    # x0
    if x0 is None:
        X0 = sample_conditions(config, n=1, seed=int(seed), dist=dist)  # (d,1)
        x0 = X0[:, 0]
    x0 = np.asarray(x0, dtype=float).reshape(-1)

    xi = np.asarray(config.xi).reshape(-1)
    d = int(config.n_states)
    if xi.size != d:
        raise ValueError(f"Mismatch: xi has {xi.size} points but state has {d} entries")

    times = np.asarray(times, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("times must be non-empty")
    if np.any(times < 0):
        raise ValueError("times must be >= 0")

    t_end = float(np.max(times))
    t_eval = np.unique(np.sort(times))

    # events
    if events == "auto":
        events = config.ocp.make_integration_events() if hasattr(config.ocp, "make_integration_events") else None

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.grid(True, alpha=0.3)

    # greys for time shading (same within each controller)
    cmap_t = plt.get_cmap(time_cmap)
    if t_eval.size == 1:
        shades = [0.6]
    else:
        shades = np.linspace(0.25, 0.85, t_eval.size)  # light -> dark

    for k, (name, ctrl) in enumerate(controllers):
        # simulate only at the snapshot times
        t, X, _status = sim_closed_loop(
            config.ocp.dynamics,
            config.ocp.closed_loop_jacobian,
            ctrl,
            tspan=[0.0, t_end],
            X0=x0,
            t_eval=t_eval,
            events=events,
            solver=solver,
        )  # X: (d, len(t_eval))

        base_color = colors[k % len(colors)]

        # plot each snapshot (same base color, different alpha/grey shade)
        for j, tj in enumerate(t_eval):
            xj = X[:, j]
            shade = float(shades[j])
            # blend towards black via grey shade but keep controller identity via base_color
            ax.plot(
                xi, xj,
                color=base_color,
                alpha=shade,
                lw=float(lw),
                label=(f"{name}, t={tj:g}" if j == 0 else None),
            )

        # add a legend entry for the controller color (without time spam)
        ax.plot([], [], color=base_color, lw=float(lw), label=str(name))

    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$x(t,\xi)$")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("State profile snapshots")
    ax.legend(loc="best", frameon=True)

    _save_fig(fig, savepath)
    return fig



def plot_space_time_heatmap(
    *,
    config,
    controller,                 # controller with eval_U(X)
    x0=None,                     # (d,) or (d,1); if None, sampled
    seed: int = 0,
    dist=None,
    tspan=(0.0, 30.0),
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
    """
    Space-time heatmap of Burgers state x(t,xi):
      x-axis: time t
      y-axis: spatial coordinate xi
      color: x
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sampling import sample_conditions

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

    _save_fig(fig, savepath)
    return fig

def plot_basin_2d(
    *,
    config,
    controller,                      # LQR / NN ctrl / or None for uncontrolled
    dir1,                            # (d,) direction
    dir2,                            # (d,) direction
    a_lim: float = 2.0,
    b_lim: float = 2.0,
    res: int = 2,
    eps: float = 1e-2,
    normalize_dirs: bool = True,
    T: float | None = None,          # if None -> uses evaluation.rollout_success (uses config.t1_initial)
    solver: str = "LSODA",
    events="auto",
    colors=("#56B4E9", "#7B2CBF"),   # success, fail (light blue, purple)
    title: str | None = None,
    figsize=(4.8, 4.0),
    savepath=None,
    return_data: bool = False,
):
    """
    Basin of attraction on a 2D slice: x0(a,b) = a*dir1 + b*dir2.

    Success is defined as: reach ||x(t)|| <= eps before blowing up / failing integration.
    - If T is None: uses evaluation.rollout_success(config, controller, x0, eps=eps).
    - If T is given: runs sim_closed_loop up to T and checks if ||x(t)|| <= eps at any sampled time.

    Returns fig (and optionally dict with grid + axes).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    d = int(config.n_states)
    dir1 = np.asarray(dir1, dtype=float).reshape(-1)
    dir2 = np.asarray(dir2, dtype=float).reshape(-1)
    if dir1.size != d or dir2.size != d:
        raise ValueError(f"dir1 and dir2 must have shape ({d},)")

    if normalize_dirs:
        dir1 = dir1 / (np.linalg.norm(dir1) + 1e-12)
        dir2 = dir2 / (np.linalg.norm(dir2) + 1e-12)

    # controller=None => uncontrolled
    ctrl = _ZeroController(config) if (controller is None) else controller

    # evaluation method
    if T is None:
        from evaluation import rollout_success
        def _is_success(x0):
            ok, _tconv = rollout_success(config, ctrl, x0, eps=float(eps))
            return bool(ok)
    else:
        from simulation import sim_closed_loop
        if events == "auto":
            events = config.ocp.make_integration_events() if hasattr(config.ocp, "make_integration_events") else None
        t_eval = np.linspace(0.0, float(T), 200)  # cheap sampling; adjust if you want
        def _is_success(x0):
            t, X, status = sim_closed_loop(
                config.ocp.dynamics, config.ocp.closed_loop_jacobian, ctrl,
                tspan=[0.0, float(T)], X0=np.asarray(x0, dtype=float),
                t_eval=t_eval, events=events, solver=solver,
                atol=1e-06, rtol=1e-03,
            )
            if status < 0 or np.isnan(X).any():
                return False
            norms = config.norm(X)
            return bool(np.any(norms <= float(eps)))

    # grid
    res = int(res)
    a = np.linspace(-float(a_lim), float(a_lim), res)
    b = np.linspace(-float(b_lim), float(b_lim), res)
    G = np.zeros((res, res), dtype=int)  # 1=success, 0=fail

    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            x0 = ai * dir1 + bj * dir2
            G[i, j] = 1 if _is_success(x0) else 0

    # plot
    cmap = ListedColormap([colors[1], colors[0]])  # index 0 fail, 1 success
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(
        G.T, origin="lower", aspect="auto",
        extent=[a[0], a[-1], b[0], b[-1]],
        cmap=cmap, interpolation="nearest", vmin=0, vmax=1,
    )
    ax.set_xlabel(r"$a$ (direction 1)")
    ax.set_ylabel(r"$b$ (direction 2)")
    ax.set_title(title or "Basin of attraction (2D slice)")

    ax.legend(
        handles=[
            Patch(facecolor=colors[0], edgecolor="none", label="success"),
            Patch(facecolor=colors[1], edgecolor="none", label="fail"),
        ],
        loc="upper right",
        frameon=True,
    )

    _save_fig(fig, savepath)

    if return_data:
        return fig, {"a": a, "b": b, "success_grid": G}
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
    import numpy as np
    import matplotlib.pyplot as plt
    from sampling import sample_conditions

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

    _save_fig(fig, savepath)
    return fig

def plot_training_losses(
    *,
    iters,
    losses,
    label: str = "loss",
    logy: bool = True,
    color: str = "#56B4E9",
    title: str | None = None,
    xlabel: str = "iteration",
    ylabel: str = "loss",
    figsize=(6.2, 3.2),
    savepath=None,
    # NEW:
    smooth: str | None = "ema",     # None | "ema" | "ma"
    ema_alpha: float = 0.03,        # for EMA
    ma_window: int = 200,           # for moving average
):
    import numpy as np
    import matplotlib.pyplot as plt

    iters = np.asarray(iters).reshape(-1)
    losses = np.asarray(losses).reshape(-1)

    if smooth is None:
        it_plot, y_plot = iters, losses

    elif smooth == "ema":
        y = np.empty_like(losses, dtype=float)
        y[0] = float(losses[0])
        a = float(ema_alpha)
        for i in range(1, len(losses)):
            y[i] = a * float(losses[i]) + (1.0 - a) * y[i - 1]
        it_plot, y_plot = iters, y

    elif smooth == "ma":
        w = int(ma_window)
        if w < 2 or w > len(losses):
            raise ValueError("ma_window must be in [2, len(losses)]")
        y = np.convolve(losses, np.ones(w) / w, mode="valid")
        it_plot = iters[w - 1 :]
        y_plot = y

    else:
        raise ValueError("smooth must be None, 'ema', or 'ma'")

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(it_plot, y_plot, color=color, lw=1.8, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save_fig(fig, savepath)
    return fig