import numpy as np
import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset
from controls.nn import GradNet
from types import SimpleNamespace
import itertools

@dataclass
class TrainConfig:
    sup_epochs: int = 1
    sup_n_steps: int = None
    sup_lr: float = 1e-5 # 
    unsup_epochs: int = 5
    unsup_n_steps: int = 30 # burgers: 30, allen_cahn: 100
    unsup_lr: float = 5e-4 # 
    batch_size: int = None
    device: str = "cpu"
    log_every: int = 1
    grad_clip: float | None = None
    # Adaptive batch sizing
    bs_max: int = 256
    adaptive_init_frac: float = 1.0           # initial batch = ceil(nominal * frac)
    adaptive_bs_max_factor: float = 1.0        # max batch = factor * initial batch (adaptive=True)
    adaptive_D0: float = 0.0
    adaptive_C: float = 0.0002
    adaptive_M: float = 1.0                  # batch growth multiplier
    # Rollout horizon
    horizon: int = 250  # burgers: 250, allen_cahn: 25
    # Rollout integration (unsupervised HJB rollout)
    dt_min: float = 1e-8
    dt_max: float = 0.1  # burgers: 0.1, allen_cahn: 0.05
    convergence_threshold: float = 0.01  # burgers: 0.0005, allen_cahn: tune per system
    # Loss weighting
    lambda_sup_base: float = 0.5            # Base supervised penalty weight (decays to 0)
    sup_err_scale: float = 10.0             # Scale factor for supervised MSE to match HJB magnitude
    # Adaptive sampling method
    adaptive_sampling: str = "gradient"     # "gradient" (gradient residual) or "hjb" (HJB residual norm)
    n_candidates: int | None = None


from controls.model_factory import save_model, load_gradnet, train_or_load_gradnet



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
    supervision: bool | None = None,
    sup_loader=None,
    sup_every: int = 50,
    val_loader=None,
):
    import numpy as np

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

    if mode == "supervised":
        if loader is None:
            raise ValueError("supervised mode requires `loader`")

        for ep in range(1, cfg.epochs + 1):
            model.train()

            sum_total = 0.0
            sum_hjb = 0.0
            sum_dpc = 0.0
            sum_sup = 0.0
            n = 0

            for batch in loader:
                batch = tuple(b.to(cfg.device) for b in batch)

                # Normalize lambdas: supervised pretraining uses lambda_hjb=0.0, lambda_sup=1.0
                lambda_hjb_norm, lambda_sup_norm = normalize_lambdas(0.0, 1.0)
                loss, hjb_err, dpc_err, sup_err = loss_fn(
                    model, batch, 
                    mode="supervised", 
                    supervision=supervision if supervision is not None else False,
                    lambda_hjb=lambda_hjb_norm,
                    lambda_sup=lambda_sup_norm,
                    sup_err_scale=cfg.sup_err_scale
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

                bs = batch[0].shape[0]
                sum_total += float(loss.item()) * bs
                sum_hjb   += float(hjb_err.item()) * bs
                sum_dpc   += float(dpc_err.item()) * bs
                sum_sup   += float(sup_err.item()) * bs
                n += bs

                step += 1
                iters.append(step)
                losses.append(float(loss.item()))
                phases.append("sup")  # Mark all supervised iterations


            # NEW: Compute validation MSE
            val_mse = None
            if val_loader is not None:
                model.eval()
                val_mse_sum = 0.0
                val_n = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = tuple(b.to(cfg.device) for b in batch)
                        if len(batch) >= 2 and batch[1] is not None:
                            X_val, G_val = batch[0], batch[1]
                            pred = model(X_val)
                            mse = nn.functional.mse_loss(pred, G_val, reduction='sum')
                            val_mse_sum += float(mse.item())
                            val_n += X_val.shape[0]
                if val_n > 0:
                    val_mse = val_mse_sum / val_n
                    val_mses.append(val_mse)
                model.train()

            if (ep % cfg.log_every) == 0:
                bs_disp = "None" if cfg.batch_size is None else str(int(cfg.batch_size))
                val_str = f" | val_mse={val_mse:.2e}" if val_mse is not None else ""
                print(f"epoch {ep:04d} | loss={sum_total/n:.2e} | hjb={sum_hjb/n:.2e} | sup={sum_sup/n:.2e} | bs={bs_disp}{val_str}")

        history = {"iters": np.asarray(iters), "loss": np.asarray(losses), "phase": np.asarray(phases)}
        if val_mses:
            history["val_mse"] = np.asarray(val_mses)
        return model, history

    if config is None:
        raise ValueError("unsupervised mode requires `config`")

    from sampling import sample_conditions, adaptive_sample_conditions

    # Get adaptive sampling method from config or default to "gradient"
    adaptive_sampling = getattr(cfg, 'adaptive_sampling', 'gradient')

    if adaptive:
        if cfg.batch_size is None:
            raise ValueError("adaptive=True requires cfg.batch_size to be set")
        _nominal_bs = int(cfg.batch_size)
        _init_bs = max(1, int(np.ceil(_nominal_bs * float(cfg.adaptive_init_frac))))
        cfg.batch_size = _init_bs
        _bs_max_local = int(float(cfg.adaptive_bs_max_factor) * _init_bs)
    else:
        _bs_max_local = int(cfg.bs_max)

    def sample_X():
        if not adaptive:
            return sample_conditions(config, cfg.batch_size)  # (d,B)

        n_cand = getattr(cfg, "n_candidates", None)
        if n_cand is None:
            raise ValueError("TrainConfig.n_candidates must be set by the experiment (e.g. n_candidates=200)")
        X_np, _ = adaptive_sample_conditions(config, cfg.batch_size, controller=model, device=cfg.device, n_candidates=n_cand)
        return X_np

    sup_iter = itertools.cycle(sup_loader) if (sup_loader is not None) else None

    # Compute total steps for smooth decay
    total_steps = cfg.epochs * cfg.unsup_n_steps
    global_step = 0
    lambda_hjb_base = 1.0  # Base HJB weight (always 1.0)
    
    # Initialize phases list for unsupervised mode
    phases = []

    for ep in range(1, cfg.epochs + 1):

        current_lr = cfg.unsup_lr / ep**2
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        model.train()
        sum_total = 0.0
        sum_hjb = 0.0
        sum_dpc = 0.0
        sum_sup = 0.0
        n = 0
        gnorms = []

        for step_i in range(cfg.unsup_n_steps):
            # Compute lambda_sup: smooth linear decay from lambda_sup_base to 0
            progress = global_step / total_steps  # 0 to 1 over all steps
            lambda_sup_current = cfg.lambda_sup_base * (1.0 - progress)
            lambda_sup_current = max(0.0, lambda_sup_current)
            
            # Normalize lambdas so they sum to 1.0
            lambda_hjb_norm, lambda_sup_norm = normalize_lambdas(lambda_hjb_base, lambda_sup_current)

            X_np = sample_X()
            X = torch.tensor(X_np.T, dtype=torch.float32, device=cfg.device)

            # unsupervised loop - use normalized lambdas
            loss, hjb_err, dpc_err, sup_err = loss_fn(
                model, (X,), mode=mode, supervision=False,
                horizon=cfg.horizon, lambda_hjb=lambda_hjb_norm,
                sup_err_scale=cfg.sup_err_scale,
                dt_min=getattr(cfg, "dt_min", 1e-8),
                dt_max=getattr(cfg, "dt_max", 0.1),
                convergence_threshold=getattr(cfg, "convergence_threshold", 0.01),
            )

            # optional supervised injection with decaying lambda_sup
            if supervision and (sup_iter is not None) and ((sup_every <= 1) or (step_i % int(sup_every) == 0)):
                Xs, Gs = next(sup_iter)
                Xs = Xs.to(cfg.device); Gs = Gs.to(cfg.device)
                # Get supervised MSE error (HJB already computed in main loss)
                _, _, _, sup_only = loss_fn(
                    model, (Xs, Gs), mode=mode, supervision=True,
                    horizon=cfg.horizon, lambda_hjb=lambda_hjb_norm, lambda_sup=lambda_sup_norm,
                    sup_err_scale=cfg.sup_err_scale,
                    dt_min=getattr(cfg, "dt_min", 1e-8),
                    dt_max=getattr(cfg, "dt_max", 0.1),
                    convergence_threshold=getattr(cfg, "convergence_threshold", 0.01),
                )
                # Only add the supervised MSE penalty: lambda_sup_norm * sup_err_scale * sup_err
                sup_penalty = lambda_sup_norm * cfg.sup_err_scale * sup_only
                loss = loss + sup_penalty
                sup_err = sup_err + sup_only

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            with torch.no_grad():
                s2 = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        s2 += float(p.grad.detach().pow(2).sum().cpu())
                gnorms.append(s2 ** 0.5)

            step += 1
            global_step += 1  # Increment global step counter
            iters.append(step)
            losses.append(float(loss.item()))
            phases.append("unsup")  # Mark all unsupervised iterations
            bs = X.shape[0]
            sum_total += float(loss.item()) * bs
            sum_hjb   += float(hjb_err.item()) * bs
            sum_dpc   += float(dpc_err.item()) * bs
            sum_sup   += float(sup_err.item()) * bs
            n += bs

        g = np.asarray(gnorms, dtype=float)
        mu = float(np.mean(g)) if g.size else 0.0
        var = float(np.var(g)) if g.size else 0.0
        heuristic = var / ((float(cfg.adaptive_C) ** 2) * (mu**2 + 1e-12))

        if adaptive and (heuristic > float(cfg.adaptive_D0)):
            cfg.batch_size = min(int(cfg.batch_size * float(cfg.adaptive_M)), _bs_max_local)

        val_mse = None
        if val_loader is not None:
            model.eval()
            val_mse_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = tuple(b.to(cfg.device) for b in batch)
                    if len(batch) >= 2 and batch[1] is not None:
                        X_val, G_val = batch[0], batch[1]
                        pred = model(X_val)
                        mse = nn.functional.mse_loss(pred, G_val, reduction='sum')
                        val_mse_sum += float(mse.item())
                        val_n += X_val.shape[0]
            if val_n > 0:
                val_mse = val_mse_sum / val_n
                val_mses.append(val_mse)
            model.train()

        if (ep % cfg.log_every) == 0:
            bs_disp = "None" if cfg.batch_size is None else str(int(cfg.batch_size))
            val_str = f" | val_mse={val_mse:.2e}" if val_mse is not None else ""
            print(f"epoch {ep:04d} | loss={sum_total/n:.2e} | hjb={sum_hjb/n:.2e} | sup={sum_sup/n:.2e} | bs={bs_disp}{val_str}")

    history = {"iters": np.asarray(iters), "loss": np.asarray(losses), "phase": np.asarray(phases)}
    if val_mses:
        history["val_mse"] = np.asarray(val_mses)
    return model, history



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

def normalize_lambdas(lambda_hjb: float, lambda_sup: float):
    """
    Normalize lambdas so they sum to 1.0.
    If lambda_hjb = 0, then lambda_sup = 1.0.
    Otherwise: lambda_hjb_norm = lambda_hjb / (lambda_hjb + lambda_sup)
               lambda_sup_norm = lambda_sup / (lambda_hjb + lambda_sup)
    """
    total = lambda_hjb + lambda_sup
    if total == 0.0:
        return 0.0, 0.0
    if lambda_hjb == 0.0:
        return 0.0, 1.0
    return lambda_hjb / total, lambda_sup / total

def loss_unified(
    model,
    batch,
    *,
    mode: str,
    supervision: bool,
    lambda_hjb: float = 1.0,
    lambda_dpc: float = 0.0,  # Not used anymore
    lambda_sup: float = 0.0,  # Supervised penalty weight (normalized to sum with lambda_hjb to 1.0)
    sup_err_scale: float = 1.0,  # Scale factor for supervised MSE to match HJB magnitude
    horizon: int = None,
    dt_min: float = 1e-8,
    dt_max: float = 0.1,
    convergence_threshold: float = 0.01,
):
    X = batch[0]
    zero = X.new_tensor(0.0)

    hjb_err = zero
    dpc_err = zero  # Always zero (DPC cost removed)
    sup_err = zero

    dVdX, _, f, L = hjb_parts(model, X)
    H = L + (dVdX * f).sum(dim=1)
    hjb_err = (H ** 2).mean()

    # Compute supervised MSE error if we have ground truth gradients
    # Note: sup_err is returned unscaled; scaling is applied via lambda_sup in loss computation
    if (len(batch) >= 2) and (batch[1] is not None):
        G = batch[1]
        pred = model(X)
        sup_err = nn.functional.mse_loss(pred, G)

    # Compute HJB error along rollout (unsupervised mode only, no supervision)
    if (mode == "unsupervised") and (not supervision) and (horizon is not None):
        x = X
        hjb_rollout_err = X.new_zeros(X.shape[0])
        converged = X.new_zeros(X.shape[0], dtype=torch.bool)

        # Get target state X_bar
        X_bar = torch.tensor(model.config.ocp.X_bar, dtype=X.dtype, device=X.device).T
        if X_bar.ndim == 1:
            X_bar = X_bar.unsqueeze(0)

        # dt parameters: small when far, large when close (from cfg or kwargs)
        for step in range(horizon):
            # Compute distance to target X_bar
            x_err = x - X_bar
            x_dist = torch.norm(x_err, dim=1)
            
            # Check convergence
            newly_converged = (x_dist < convergence_threshold) & (~converged)
            converged = converged | newly_converged
            
            if converged.all():
                break
            
            # Adaptive dt based on distance to target
            # Adaptive dt based on distance to target
            max_dist = 2.0
            normalized_dist = torch.clamp(x_dist / max_dist, 0.0, 1.0)
            current_dt = dt_min * (dt_max / dt_min) ** (1.0 - normalized_dist)
            
            # Compute HJB error at this rollout step
            dVdX, _, f, L = hjb_parts(model, x)
            
            # Ensure f has correct shape (B, n) - fix if it's transposed
            if f.shape != x.shape:
                if f.shape == x.shape[::-1]:  # If f is (n, B) instead of (B, n)
                    f = f.t()
                else:
                    raise RuntimeError(f"Shape mismatch: x has shape {x.shape}, f has shape {f.shape}")
            
            H = L + (dVdX * f).sum(dim=1)
            
            # Accumulate HJB error (only for non-converged trajectories)
            active_mask = ~converged
            hjb_rollout_err[active_mask] = hjb_rollout_err[active_mask] + (H[active_mask] ** 2) * current_dt[active_mask]
            
            # Update state - reshape current_dt for proper broadcasting
            current_dt_expanded = current_dt.unsqueeze(1)  # (B, 1) for broadcasting with (B, n)
            x = x + f * current_dt_expanded

        # Add HJB error from rollout to initial HJB error
        actual_steps = step + 1 if step < horizon else horizon
        if actual_steps > 0:
            hjb_err = hjb_err + hjb_rollout_err.mean() / actual_steps

    total = (lambda_hjb * hjb_err) + (lambda_dpc * dpc_err) + (lambda_sup * sup_err_scale * sup_err)
    return total, hjb_err, dpc_err, sup_err