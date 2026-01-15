from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

from sampling import sample_conditions
from simulation import sim_closed_loop


# ---------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------

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
    for k, (name, ctrl) in enumerate(_iter_named_controllers(controllers)):
        V = _value_from_controller_or_gradnet(ctrl, X, device=device, n_quad=n_quad)
        ax.plot(s, V, label=str(name))



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
    for k, (name, ctrl) in enumerate(_iter_named_controllers(controllers)):
        V = _value_from_controller_or_gradnet(ctrl, X, device=device, n_quad=n_quad)



        # in plot_value_vs_state_norm:
        ax.scatter(xnorm, V, s=float(s), alpha=float(alpha), label=str(name))

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

    _save_fig(fig, savepath)

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
    colors=None,
    title: str | None = None,
    figsize=(6.2, 3.2),
    savepath=None,
):
    """
    1D "truth plot": shock intensity s vs HJB residual error at x = s * shock.

    Notes:
      - This calls `controls.train.loss_unified(...)` with mode="unsupervised", supervision=False.
      - That returns (total, hjb_err, dpc_err, sup_err); we plot hjb_err which equals H(x)^2.
    """
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
        # If it's an LQR controller, wrap as model.
        if hasattr(obj, "eval_dVdX"):
            return _LQRAsGradModel(config)

        # If it's a policy wrapper (Control), unwrap to its grad_net if present.
        if hasattr(obj, "grad_net") and (obj.grad_net is not None):
            obj = obj.grad_net

        # assume it's a torch model like GradNet with .config
        if not hasattr(obj, "config"):
            raise ValueError(
                "For non-LQR, pass the GradNet itself (e.g. ctrl.grad_net or model), "
                "not a string or a controller without gradient information."
            )
        return obj

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for k, (name, obj) in enumerate(_iter_named_controllers(controllers)):
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

        ax.plot(s_grid, ys, label=str(name))

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

    for k, (name, ctrl) in enumerate(_iter_named_controllers(controllers)):
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

        ax.plot(t, V, label=str(name))

        if mark_bumps and V.size >= 2:
            dV = np.diff(V)
            bump_idx = np.where(dV > float(bump_tol))[0] + 1
            if bump_idx.size > 0:
                ax.plot(t[bump_idx], V[bump_idx], "o", ms=3.5, alpha=0.9)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$V(x(t))$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)

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

    _save_fig(fig, savepath)
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

    _save_fig(fig, savepath)
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
    # NEW: multi-series support (preferred)
    series=None,  # list of (name, iters, losses)
    # NEW:
    smooth: str | None = "ema",     # None | "ema" | "ma"
    ema_alpha: float = 0.03,        # for EMA
    ma_window: int = 200,           # for moving average
):
    if series is None:
        if iters is None or losses is None:
            raise ValueError("Provide either (iters, losses) or series=[(name,iters,losses), ...].")
        series = [(label, iters, losses)]

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

    # Align phases on a shared timeline:
    # - supervised segments start at 1
    # - unsupervised-only runs can be shifted by the maximum supervised length across series
    sup_lengths = []
    for name_k, it_k, _lo_k, ph_k in norm_series:
        if it_k is None:
            continue
        it_k = np.asarray(it_k).reshape(-1)
        if ph_k is not None:
            ph = np.asarray(ph_k).reshape(-1)
            if ph.size == it_k.size:
                sup_lengths.append(int(np.sum(ph == "sup")))
        elif "supervised" in str(name_k).lower() or str(name_k).lower().strip() == "sup":
            sup_lengths.append(int(it_k.size))
    global_k = int(max(sup_lengths)) if sup_lengths else 0

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for k, (name_k, it_k, lo_k, ph_k) in enumerate(norm_series):
        it_k = np.asarray(it_k).reshape(-1)
        lo_k = np.asarray(lo_k).reshape(-1)

        # optional alignment: shift pure-unsupervised runs by global supervised length
        if global_k > 0 and ph_k is None:
            nm = str(name_k).lower()
            is_unsup = ("unsup" in nm) or ("unsupervised" in nm)
            is_sup = ("supervised" in nm) or (nm.strip() == "sup")
            if is_unsup and (not is_sup) and (it_k.size > 0) and int(it_k[0]) == 1:
                it_k = it_k + global_k

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
    _save_fig(fig, savepath)
    return fig