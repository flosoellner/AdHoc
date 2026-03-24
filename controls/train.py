import numpy as np
import os
import torch
import torch.nn as nn
from dataclasses import dataclass, replace
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from controls.nn import GradNet
import itertools

@dataclass
class TrainConfig:
    sup_epochs: int = 1
    sup_lr: float = 1e-4
    rollouts: int = 5
    unsup_lr: float = 1e-3
    batch_size: int = None
    device: str = "cpu"
    log_every: int = 1
    grad_clip: Optional[float] = None
    horizon: int = 250
    dt_min: float = 1e-2
    dt_max: float = 1
    convergence_threshold: float = 1e-2
    lambda_sup_base: float = 1.0
    n_candidates: Optional[int] = None
    # Set by train() wrapper to select sup/unsup phase
    epochs: Optional[int] = None
    lr: Optional[float] = None


def to_tensor_X(dataset, key="X"):
    X = dataset.get(key, dataset.get(key.lower()))
    if X is None:
        raise KeyError(f"dataset must contain {key} (or {key.lower()})")
    X = np.asarray(X)
    if X.ndim == 2 and X.shape[0] < X.shape[1]:
        X = X.T  # (N,d)
    return torch.tensor(X, dtype=torch.float32)


def make_loader_X(dataset, cfg: TrainConfig, shuffle=True):
    X = to_tensor_X(dataset, key="X")
    return DataLoader(TensorDataset(X), batch_size=cfg.batch_size, shuffle=shuffle, drop_last=False)


def make_loader_XG(dataset, cfg: TrainConfig, shuffle=True):
    X = to_tensor_X(dataset, key="X")
    G = dataset.get("dVdX", dataset.get("grad"))
    if G is None:
        raise KeyError("dataset must contain dVdX or grad")
    G = np.asarray(G)
    if G.ndim == 2 and G.shape[0] < G.shape[1]:
        G = G.T  # (N,d)
    G = torch.tensor(G, dtype=torch.float32)
    return DataLoader(TensorDataset(X, G), batch_size=cfg.batch_size, shuffle=shuffle, drop_last=False)



def _compute_val_mse(model, val_loader, device):
    """Compute mean validation MSE. Returns float or None."""
    if val_loader is None:
        return None
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = tuple(b.to(device) for b in batch)
            if len(batch) >= 2 and batch[1] is not None:
                pred = model(batch[0])
                total += float(nn.functional.mse_loss(pred, batch[1], reduction='sum').item())
                n += batch[0].shape[0]
    model.train()
    return total / n if n > 0 else None


def _log_epoch(ep, sum_total, sum_hjb, sum_sup, n, cfg, val_mse=None, conv_pct=None, dt_range=None):
    """Print one-line supervised epoch summary (after all batches in the epoch)."""
    if (ep % cfg.log_every) != 0:
        return
    bs_disp = "None" if cfg.batch_size is None else str(int(cfg.batch_size))
    val_str = f" | val_mse={val_mse:.2e}" if val_mse is not None else ""
    dt_str = f" | dt=[{dt_range[0]:.2e},{dt_range[1]:.2e}]" if dt_range is not None else ""
    print(f"epoch {ep:04d} | loss={sum_total/n:.2e} | hjb={sum_hjb/n:.2e} | sup={sum_sup/n:.2e} | bs={bs_disp}{val_str}{dt_str}")


def _log_rollout(rollout_idx, n_rollouts, sum_total, sum_hjb, sum_sup, n, cfg, val_mse=None, conv_pct=None, dt_range=None):
    """Print one line per unsupervised rollout (not per trajectory timestep)."""
    if (rollout_idx % cfg.log_every) != 0:
        return
    bs_disp = "None" if cfg.batch_size is None else str(int(cfg.batch_size))
    val_str = f" | val_mse={val_mse:.2e}" if val_mse is not None else ""
    dt_str = f" | dt=[{dt_range[0]:.2e},{dt_range[1]:.2e}]" if dt_range is not None else ""
    conv_str = f" | conv={conv_pct:.2%}" if conv_pct is not None else ""
    print(
        f"rollout {rollout_idx:04d}/{n_rollouts:04d} | loss={sum_total/n:.2e} | hjb={sum_hjb/n:.2e} | "
        f"sup={sum_sup/n:.2e} | bs={bs_disp}{val_str}{conv_str}{dt_str}"
    )


def _build_history(iters, losses, phases, val_mses, batch_size=1, mean_dts=None):
    """Assemble the history dict returned by train_loop."""
    history = {"iters": np.asarray(iters), "loss": np.asarray(losses), "phase": np.asarray(phases),
               "batch_size": int(batch_size) if batch_size is not None else 1}
    if val_mses:
        history["val_mse"] = np.asarray(val_mses)
    if mean_dts is not None:
        history["mean_dt"] = np.asarray(mean_dts, dtype=float)
    return history


def train_loop(
    model,
    loader,
    loss_fn,
    cfg: TrainConfig,
    *,
    optimizer=None,
    mode: str = None,
    config=None,
    adaptive: bool = False,
    supervision: Optional[bool] = None,
    sup_loader=None,
    val_loader=None,
):
    model = model.to(cfg.device) if hasattr(model, "to") else model
    
    if mode not in {"supervised", "unsupervised"}:
        raise ValueError("mode must be 'supervised' or 'unsupervised'")
    
    # Initialize optimizer with the correct learning rate based on mode
    if optimizer is None:
        if mode == "supervised":
            lr = getattr(cfg, 'sup_lr', getattr(cfg, 'lr', 5e-4))
        else:  # unsupervised
            lr = getattr(cfg, 'unsup_lr', getattr(cfg, 'lr', 1e-4))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    iters, losses, phases, step = [], [], [], 0
    val_mses = []
    mean_dts = []

    if mode == "supervised":
        if loader is None:
            raise ValueError("supervised mode requires `loader`")

        for ep in range(1, cfg.epochs + 1):
            model.train()

            sum_total = 0.0
            sum_hjb = 0.0
            sum_sup = 0.0
            n = 0

            for batch in loader:
                batch = tuple(b.to(cfg.device) for b in batch)

                loss, hjb_err, sup_err, _, _, _ = loss_fn(
                    model, batch,
                    mode="supervised",
                    supervision=supervision if supervision is not None else False,
                    lambda_hjb=0.0,
                    lambda_sup=1.0,
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

                bs = batch[0].shape[0]
                sum_total += float(loss.item()) * bs
                sum_hjb   += float(hjb_err.item()) * bs
                sum_sup   += float(sup_err.item()) * bs
                n += bs

            # One history point per epoch (mean over batches), not per batch / trajectory step
            step += 1
            iters.append(step)
            losses.append(sum_total / n if n > 0 else 0.0)
            phases.append("sup")
            mean_dts.append(float("nan"))

            val_mse = _compute_val_mse(model, val_loader, cfg.device)
            if val_mse is not None:
                val_mses.append(val_mse)

            _log_epoch(ep, sum_total, sum_hjb, sum_sup, n, cfg, val_mse)

        return model, _build_history(iters, losses, phases, val_mses, cfg.batch_size, mean_dts)

    if config is None:
        raise ValueError("unsupervised mode requires `config`")

    from sampling import sample_conditions, adaptive_sample_conditions

    def sample_X():
        if not adaptive:
            return sample_conditions(config, cfg.batch_size)  # (d,B)

        n_cand = getattr(cfg, "n_candidates", None)
        if n_cand is None:
            raise ValueError("TrainConfig.n_candidates must be set by the experiment (e.g. n_candidates=200)")
        X_np, _ = adaptive_sample_conditions(config, cfg.batch_size, controller=model, device=cfg.device, n_candidates=n_cand)
        return X_np

    sup_iter = itertools.cycle(sup_loader) if (sup_loader is not None) else None

    total_steps = cfg.epochs
    global_step = 0

    phases = []
    rollout_kw = dict(horizon=cfg.horizon, dt_min=cfg.dt_min, dt_max=cfg.dt_max,
                      convergence_threshold=cfg.convergence_threshold)

    for ep in range(1, cfg.epochs + 1):

        current_lr = cfg.unsup_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        model.train()
        sum_total = 0.0
        sum_hjb = 0.0
        sum_sup = 0.0
        sum_conv = 0.0
        n = 0
        n_conv = 0
        dt_range_epoch = (float("inf"), float("-inf"))  # (min, max) over batch rollouts

        progress = global_step / total_steps
        lambda_sup_current = max(0.0, cfg.lambda_sup_base * (1.0 - progress))

        X_np = sample_X()
        X = torch.tensor(X_np.T, dtype=torch.float32, device=cfg.device)

        loss, hjb_err, sup_err, conv_frac, dt_range, mean_dt = loss_fn(
            model, (X,), mode=mode, supervision=False,
            lambda_hjb=1.0,
            lambda_sup=lambda_sup_current,
            **rollout_kw,
        )

        if supervision and (sup_iter is not None):
            Xs, Gs = next(sup_iter)
            Xs = Xs.to(cfg.device); Gs = Gs.to(cfg.device)
            _, _, sup_only, _, _, _ = loss_fn(
                model, (Xs, Gs), mode=mode, supervision=True,
                lambda_hjb=1.0, lambda_sup=lambda_sup_current,
                **rollout_kw,
            )
            loss = loss + lambda_sup_current * sup_only
            sup_err = sup_err + sup_only

        if conv_frac is not None:
            sum_conv += conv_frac * X.shape[0]
            n_conv += X.shape[0]

        if dt_range is not None:
            dmn, dmx = dt_range
            dt_range_epoch = (
                min(dt_range_epoch[0], dmn),
                max(dt_range_epoch[1], dmx),
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        step += 1
        global_step += 1
        iters.append(step)
        losses.append(float(loss.item()))
        phases.append("unsup")
        mean_dts.append(float(mean_dt) if mean_dt is not None else float("nan"))
        bs = X.shape[0]
        sum_total += float(loss.item()) * bs
        sum_hjb   += float(hjb_err.item()) * bs
        sum_sup   += float(sup_err.item()) * bs
        n += bs

        val_mse = _compute_val_mse(model, val_loader, cfg.device)
        if val_mse is not None:
            val_mses.append(val_mse)

        conv_pct = sum_conv / n_conv if n_conv > 0 else None
        dt_range_final = dt_range_epoch if dt_range_epoch[0] <= dt_range_epoch[1] else None
        _log_rollout(
            ep, cfg.epochs, sum_total, sum_hjb, sum_sup, n, cfg,
            val_mse, conv_pct, dt_range_final,
        )

    return model, _build_history(iters, losses, phases, val_mses, cfg.batch_size, mean_dts)



def _physics(model, X):
    """Get torch physics module from OCP (single source: ocp.physics_module())."""
    if not hasattr(model, "_physics"):
        ocp = model.config.ocp
        model._physics = ocp.physics_module().to(device=X.device, dtype=X.dtype)
    return model._physics


def hjb_parts(model, X: torch.Tensor):
    phys = _physics(model, X)
    
    dVdX = model(X)            # The Gradient (Critic output)
    u = phys.get_control(dVdX) # The Action (Actor logic)
    f = phys.dynamics(X, u)    # The Physics (ODE right-hand side)
    L = phys.running_cost(X, u)# The Running Cost (L)
    
    return dVdX, u, f, L

def _hjb_rollout(model, X, horizon, dt_min, dt_max):
    """Forward-rollout HJB error accumulation.

    Returns (scalar mean error, frac_converged, dt_range, mean_dt).
    dt_range is (dt_min_used, dt_max_used) over all steps/samples, or None if no steps taken.
    mean_dt is the mean of per-trajectory step sizes over all integration steps (active samples only).
    """
    ocp = model.config.ocp
    x = X
    err = X.new_zeros(X.shape[0])
    converged = X.new_zeros(X.shape[0], dtype=torch.bool)
    actual_steps = 0
    dt_min_used, dt_max_used = float("inf"), float("-inf")
    sum_dt = 0.0
    n_dt = 0
    for step in range(horizon):
        dVdX, _, f, L = hjb_parts(model, x)
        if f.shape != x.shape:
            if f.shape == x.shape[::-1]:
                f = f.t()
            else:
                raise RuntimeError(f"Shape mismatch: x={x.shape}, f={f.shape}")
        converged = converged | ocp.convergence_torch(f, L, model.config.conv_tol, model.config.fp_tol)
        if converged.all():
            break
        f_norm = torch.norm(f, dim=1)
        c = dt_max
        dt_raw = c / (f_norm + 1e-8)
        dt = torch.clamp(dt_raw, dt_min, dt_max)
        dmin, dmax = float(dt.min().item()), float(dt.max().item())
        dt_min_used = min(dt_min_used, dmin)
        dt_max_used = max(dt_max_used, dmax)
        H = L + (dVdX * f).sum(dim=1)
        active = ~converged
        err[active] += (H[active] ** 2) * dt[active]
        if bool(active.any().item()):
            sum_dt += float(dt[active].sum().item())
            n_dt += int(active.sum().item())
        x = x + f * dt.unsqueeze(1)
        actual_steps = step + 1
    frac = float(converged.float().mean().item())
    hjb_mean = err.mean() / actual_steps if actual_steps > 0 else X.new_tensor(0.0)
    dt_range = (dt_min_used, dt_max_used) if actual_steps > 0 else None
    mean_dt = (sum_dt / n_dt) if n_dt > 0 else None
    return hjb_mean, frac, dt_range, mean_dt


def loss_unified(
    model,
    batch,
    *,
    mode: str,
    supervision: bool,
    lambda_hjb: float = 1.0,
    lambda_sup: float = 0.0,
    horizon: int = None,
    dt_min: float = 1e-8,
    dt_max: float = 0.1,
    convergence_threshold: float = None,
):
    X = batch[0]
    zero = X.new_tensor(0.0)
    hjb_err = zero
    sup_err = zero

    dVdX, _, f, L = hjb_parts(model, X)
    hjb_err = (L + (dVdX * f).sum(dim=1)).pow(2).mean()

    if (len(batch) >= 2) and (batch[1] is not None):
        sup_err = nn.functional.mse_loss(model(X), batch[1])

    conv_frac = None
    dt_range = None
    mean_dt = None
    if (mode == "unsupervised") and (not supervision) and (horizon is not None):
        rollout_err, conv_frac, dt_range, mean_dt = _hjb_rollout(model, X, horizon, dt_min, dt_max)
        hjb_err = hjb_err + rollout_err

    total = lambda_hjb * hjb_err + lambda_sup * sup_err
    return total, hjb_err, sup_err, conv_frac, dt_range, mean_dt


def train(model, config=None, train_config=None, data=None, val_data=None,
          mode="unsupervised", pretrain=False, supervision=None, adaptive=False):
    """Thin wrapper for model_factory: build loaders and call train_loop. Returns history dict."""
    val_loader = None
    if val_data and (val_data.get("dVdX") is not None or val_data.get("grad") is not None):
        val_loader = make_loader_XG(val_data, train_config, shuffle=False)
    sup_cfg = replace(train_config, epochs=train_config.sup_epochs, lr=train_config.sup_lr)
    unsup_cfg = replace(train_config, epochs=train_config.rollouts, lr=train_config.unsup_lr)

    # Build supervised loader for penalty injection during unsupervised training
    sup_loader = None
    if supervision and data is not None and (data.get("dVdX") is not None or data.get("grad") is not None):
        sup_loader = make_loader_XG(data, train_config, shuffle=True)

    if mode == "supervised":
        loader = make_loader_XG(data, train_config, shuffle=True)
        _, history = train_loop(model, loader, loss_unified, sup_cfg, mode="supervised", config=config, val_loader=val_loader)
        return history

    if mode == "unsupervised":
        # Optional supervised pretraining phase
        if pretrain:
            if data is None:
                raise ValueError("pretrain=True requires supervised data")
            loader = make_loader_XG(data, train_config, shuffle=True)
            _, h_pre = train_loop(model, loader, loss_unified, sup_cfg, mode="supervised", config=config, val_loader=val_loader)

        _, h_unsup = train_loop(model, None, loss_unified, unsup_cfg, mode="unsupervised",
                                config=config, adaptive=adaptive, supervision=supervision,
                                sup_loader=sup_loader, val_loader=val_loader)

        if not pretrain:
            return h_unsup

        # Merge pretraining + unsupervised histories
        off = int(h_pre["iters"][-1]) if len(h_pre["iters"]) else 0
        phase = np.array(["sup"] * len(h_pre["iters"]) + ["unsup"] * len(h_unsup["iters"]))
        history = {
            "iters": np.concatenate([h_pre["iters"], h_unsup["iters"] + off]),
            "loss": np.concatenate([h_pre["loss"], h_unsup["loss"]]),
            "phase": phase,
        }
        if "val_mse" in h_pre:
            v2 = h_unsup.get("val_mse", [])
            history["val_mse"] = np.concatenate([h_pre["val_mse"], v2]) if np.size(v2) else h_pre["val_mse"]
        m_pre = np.full(len(h_pre["iters"]), np.nan, dtype=float)
        m_un = h_unsup.get("mean_dt", np.full(len(h_unsup["iters"]), np.nan, dtype=float))
        history["mean_dt"] = np.concatenate([m_pre, np.asarray(m_un, dtype=float)])
        return history

    raise ValueError("mode must be 'supervised' or 'unsupervised'")