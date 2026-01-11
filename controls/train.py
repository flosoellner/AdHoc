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
    sup_epochs: int = 10
    sup_lr: float = 1e-3
    unsup_epochs: int = 10
    unsup_lr: float = 1e-4
    batch_size: int = None
    device: str = "cpu"
    log_every: int = 1
    grad_clip: float | None = None

# add new knobs to TrainConfig if you want:

bs_max = 256
bs_warmup_epochs = 1
D0 = 1.0
C = 1
M = 1.5

def save_model(path, *, config, grad_net=None, value_net=None, extra=None, history=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "meta": {
            "system": config.system,
            "n_states": config.n_states,
            "n_controls": config.n_controls,
            "seed": config.seed,
            "extra": extra or {},
        },
        "grad_net": grad_net.state_dict() if grad_net is not None else None,
        "value_net": value_net.state_dict() if value_net is not None else None,
    }
    history_to_save = None if history is None else {
        "iters": list(history["iters"]),
        "loss": list(history["loss"]),
    }
    ckpt["history"] = history_to_save
    torch.save(ckpt, path)


def load_gradnet(path, *, config, device="cpu"):
    from controls.nn import GradNet
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if ckpt.get("grad_net") is None:
        raise ValueError(f"No grad_net weights in checkpoint: {path}")

    grad_net = GradNet(config).to(device)
    sd = ckpt["grad_net"]
    sd = {k: v for k, v in sd.items() if not k.startswith("_physics.")}
    grad_net.load_state_dict(sd, strict=True)
    grad_net.eval()

    hist = ckpt.get("history")
    if hist is not None:
        hist = {"iters": np.asarray(hist["iters"]), "loss": np.asarray(hist["loss"])}
    return grad_net, ckpt.get("meta", {}), hist 

def train_or_load_gradnet(
    *,
    config,
    kind: str,
    use_lqr: bool,
    train_mode: str,            # "supervised" | "unsupervised" | "hybrid"
    adaptive: bool = False, 
    supervision: bool = False,
    train_cfg: TrainConfig,
    train_loader=None,
    ckpt_dir: str = "./saved_models",
    force_retrain: bool = False,
    device: str = "cpu",
    data=None,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(
        ckpt_dir,
        f"{kind}_useLQR{int(use_lqr)}_{train_mode}_adapt{int(adaptive)}_sup{int(supervision)}"
        f"_d{config.n_states}_m{config.n_controls}_seed{config.seed}.pt"
    )

    if (not force_retrain) and os.path.exists(ckpt_path):
        grad_net, meta, history = load_gradnet(ckpt_path, config=config, device=device)
        return (grad_net, meta, history)

    grad_net = GradNet(config, use_lqr=use_lqr).to(device)

    sup_cfg   = SimpleNamespace(**vars(train_cfg), epochs=train_cfg.sup_epochs,   lr=train_cfg.sup_lr)
    unsup_cfg = SimpleNamespace(**vars(train_cfg), epochs=train_cfg.unsup_epochs, lr=train_cfg.unsup_lr)

    history = None

    # before calling train_loop in unsupervised / hybrid-unsup:
    sup_loader = None
    if supervision:
        if data is None:
            raise ValueError("supervision=True requires `data` (X,dVdX)")
        sup_loader = make_loader_XG(data, train_cfg, shuffle=True)


    if train_mode == "supervised":
        if data is None:
            raise ValueError("supervised requires data")


        grad_net, history = train_loop(
            grad_net,
            make_loader_XG(data, train_cfg),
            loss_unified,
            sup_cfg,
            mode="supervised",
        )


    elif train_mode == "unsupervised":

            # unsupervised call:
            grad_net, history = train_loop(
                grad_net, None, loss_unified, unsup_cfg,
                mode="unsupervised", config=config, adaptive=adaptive,
                supervision=supervision, sup_loader=sup_loader, sup_every=1,
            )



    elif train_mode == "hybrid":
        if data is None:
            raise ValueError("hybrid requires data")


        grad_net, hist_sup = train_loop(
            grad_net,
            make_loader_XG(data, train_cfg),
            loss_unified,
            sup_cfg,
            mode="supervised",
        )
        grad_net, hist_unsup = train_loop(
            grad_net,
            None,
            loss_unified,
            unsup_cfg,
            mode="unsupervised",
            config=config,
            adaptive=adaptive,
            supervision=supervision,
            sup_loader=sup_loader,
            sup_every=1,
        )

        it_sup, lo_sup = hist_sup["iters"], hist_sup["loss"]
        it_uns, lo_uns = hist_unsup["iters"], hist_unsup["loss"]
        offset = int(it_sup[-1]) if len(it_sup) else 0

        history = {
            "iters": np.concatenate([it_sup, it_uns + offset]),
            "loss":  np.concatenate([lo_sup, lo_uns]),
            "phase": np.array((["sup"] * len(it_sup)) + (["unsup"] * len(it_uns))),
        }

    else:
        raise ValueError("train_mode must be 'supervised', 'unsupervised', or 'hybrid'")

    meta_out = {"kind": kind, "use_lqr": use_lqr, "train_mode": train_mode, "adaptive": adaptive, "supervision": supervision}
    save_model(ckpt_path, config=config, grad_net=grad_net, value_net=None, extra=meta_out, history=history)

    return (grad_net, meta_out, history)



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


# -------------------------
# Generic training loop
# -------------------------
import itertools  # make sure this exists at file top

def train_loop(
    model,
    loader,
    loss_fn,
    cfg: TrainConfig,
    *,
    optimizer=None,
    mode: str = None,
    n_steps: int = 500,
    config=None,
    adaptive: bool = False,
    supervision: bool | None = None,
    sup_loader=None,
    sup_every: int = 1,
):
    import numpy as np

    model = model.to(cfg.device) if hasattr(model, "to") else model
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if mode not in {"supervised", "unsupervised"}:
        raise ValueError("mode must be 'supervised' or 'unsupervised'")

    iters, losses, step = [], [], 0

    # -------------------------
    # supervised (EARLY RETURN)
    # -------------------------
    if mode == "supervised":
        if loader is None:
            raise ValueError("supervised mode requires `loader`")

        for ep in range(1, cfg.epochs + 1):
            model.train()


            # inside: for ep in range(1, cfg.epochs + 1):
            sum_total = 0.0
            sum_hjb = 0.0
            sum_dpc = 0.0
            sum_sup = 0.0
            n = 0

            for batch in loader:
                batch = tuple(b.to(cfg.device) for b in batch)

                loss, hjb_err, dpc_err, sup_err = loss_fn(model, batch, mode="supervised", supervision=True)

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

            if (ep % cfg.log_every) == 0:
                print(f"epoch {ep:04d} | loss={sum_total/n:.2e} | bs={int(cfg.batch_size)} | heur=nan")

        history = {"iters": np.asarray(iters), "loss": np.asarray(losses)}
        return model, history

    # -------------------------
    # unsupervised
    # -------------------------
    if config is None:
        raise ValueError("unsupervised mode requires `config`")

    from sampling import sample_conditions, adaptive_sample_conditions

    def sample_X():
        if not adaptive:
            return sample_conditions(config, cfg.batch_size)  # (d,B)
        X_np, _ = adaptive_sample_conditions(config, cfg.batch_size, controller=model)
        return X_np

    sup_iter = itertools.cycle(sup_loader) if (sup_loader is not None) else None

    for ep in range(1, cfg.epochs + 1):
        model.train()
        sum_total = 0.0
        sum_hjb = 0.0
        sum_dpc = 0.0
        sum_sup = 0.0
        n = 0
        gnorms = []

        for step_i in range(n_steps):
            X_np = sample_X()
            X = torch.tensor(X_np.T, dtype=torch.float32, device=cfg.device)

            # unsupervised loop (inside for step_i in range(n_steps):)
            loss, hjb_err, dpc_err, sup_err = loss_fn(model, (X,), mode=mode, supervision=False)

            # optional supervised injection (adds sup-only loss; no subtraction hack needed)
            if supervision and (sup_iter is not None) and ((sup_every <= 1) or (step_i % int(sup_every) == 0)):
                Xs, Gs = next(sup_iter)
                Xs = Xs.to(cfg.device); Gs = Gs.to(cfg.device)
                sup_total, _, _, sup_only = loss_fn(model, (Xs, Gs), mode=mode, supervision=True)
                loss = loss + sup_total
                sup_err = sup_err + sup_only  # so logging shows the injected supervised amount

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
            iters.append(step)
            losses.append(float(loss.item()))
            bs = X.shape[0]
            sum_total += float(loss.item()) * bs
            sum_hjb   += float(hjb_err.item()) * bs
            sum_dpc   += float(dpc_err.item()) * bs
            sum_sup   += float(sup_err.item()) * bs
            n += bs

        g = np.asarray(gnorms, dtype=float)
        mu = float(np.mean(g)) if g.size else 0.0
        var = float(np.var(g)) if g.size else 0.0
        heuristic = var / ((C**2) * (mu**2 + 1e-12))

        if adaptive and (heuristic > D0):
            cfg.batch_size = min(int(cfg.batch_size * M), int(bs_max))

        if (ep % cfg.log_every) == 0:
            print(
                f"epoch {ep:04d} | "
                f"total={sum_total/n:.2e} | hjb={sum_hjb/n:.2e} | dpc={sum_dpc/n:.2e} | sup={sum_sup/n:.2e} | "
                f"bs={int(cfg.batch_size)} | heur={heuristic:.2e}"
            )

    history = {"iters": np.asarray(iters), "loss": np.asarray(losses)}
    return model, history



from problems.burgers import BurgersPhysics
from problems.allen_cahn import AllenCahnPhysics

def _physics(model, X):
    if not hasattr(model, "_physics"):
        ocp = model.config.ocp
        if model.config.system == "burgers":
            model._physics = BurgersPhysics(ocp).to(device=X.device, dtype=X.dtype)
        elif model.config.system == "allen_cahn":
            model._physics = AllenCahnPhysics(ocp).to(device=X.device, dtype=X.dtype)
        else:
            raise ValueError(f"Unknown system: {model.config.system}")
    return model._physics


def hjb_parts(model, X: torch.Tensor):
    phys = _physics(model, X)
    
    dVdX = model(X)            # The Gradient (Critic output)
    u = phys.get_control(dVdX) # The Action (Actor logic)
    f = phys.dynamics(X, u)    # The Physics (ODE right-hand side)
    L = phys.running_cost(X, u)# The Running Cost (L)
    
    return dVdX, u, f, L

def loss_unified(
    model,
    batch,
    *,
    mode: str,                    # "supervised" | "unsupervised"
    supervision: bool,            # selects which term(s) are active
    lambda_hjb: float = 1e-6,
    lambda_dpc: float = 1e-3,
    lambda_sup: float = 1e-6,
    horizon: int = 30,
    dt: float = 0.3,
):
    X = batch[0]
    zero = X.new_tensor(0.0)

    hjb_err = zero
    dpc_err = zero
    sup_err = zero

    if (mode == "unsupervised") and (not supervision):
        # --- HJB residual ---
        dVdX, _, f, L = hjb_parts(model, X)
        H = L + (dVdX * f).sum(dim=1)
        hjb_err = (H ** 2).mean()

        # --- DPC rollout ---
        x = X
        total_cost = X.new_zeros(X.shape[0])
        for _ in range(horizon):
            _, _, f, L = hjb_parts(model, x)
            total_cost = total_cost + L * dt
            x = torch.clamp(x + f * dt, -1.5, 1.5)
        dpc_err = total_cost.mean()

    else:
        # supervised term only
        if (len(batch) >= 2) and (batch[1] is not None):
            G = batch[1]
            pred = model(X)
            sup_err = nn.functional.mse_loss(pred, G)

    total = (lambda_hjb * hjb_err) + (lambda_dpc * dpc_err) + (lambda_sup * sup_err)
    return total, hjb_err, dpc_err, sup_err