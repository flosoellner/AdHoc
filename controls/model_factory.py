import os
import numpy as np
import torch
from dataclasses import dataclass, replace, fields
from typing import Optional
from controls.nn import GradNet, Control
from controls.train import TrainConfig

# --- 1. Controller Configuration (Dataclass) ---
@dataclass
class ControllerConfig:
    """Configuration for a single controller variant."""
    enabled: bool = False
    kind: str = "gradnet"
    use_lqr: bool = True
    train_mode: str = "unsupervised"
    pretrain: bool = False
    adaptive: bool = False
    supervision: Optional[bool] = None

# --- 2. Default Configurations ---
DEFAULT_CONTROLLER_CONFIGS = {
    "GradNet":                ControllerConfig(use_lqr=False, train_mode="unsupervised"),
    "GradQRNet":              ControllerConfig(use_lqr=True,  train_mode="unsupervised"),
    "Sup GradQRNet":          ControllerConfig(use_lqr=True,  train_mode="supervised"),
    "GradQRNet (sup)":        ControllerConfig(use_lqr=True,  train_mode="unsupervised", supervision=True),
    "GradQRNet (pre)":        ControllerConfig(use_lqr=True,  train_mode="unsupervised", pretrain=True),
    "GradQRNet (pre/sup)":    ControllerConfig(use_lqr=True,  train_mode="unsupervised", pretrain=True, supervision=True),
    "GradNet (sup/ad)":       ControllerConfig(use_lqr=False, train_mode="unsupervised", adaptive=True, supervision=True),
    "GradQRNet (ad)":         ControllerConfig(use_lqr=True,  train_mode="unsupervised", adaptive=True),
    "GradQRNet (sup/ad)":     ControllerConfig(use_lqr=True,  train_mode="unsupervised", adaptive=True, supervision=True),
    "GradQRNet (pre/ad)":     ControllerConfig(use_lqr=True,  train_mode="unsupervised", pretrain=True, adaptive=True),
    "GradQRNet (pre/sup/ad)": ControllerConfig(use_lqr=True,  train_mode="unsupervised", pretrain=True, adaptive=True, supervision=True),
}

def get_default_controller_configs():
    """Returns a deep copy of defaults to prevent mutable state bugs."""
    return {k: replace(v) for k, v in DEFAULT_CONTROLLER_CONFIGS.items()}


# Keys that belong to ControllerConfig vs TrainConfig (for splitting overrides)
_CONTROLLER_KEYS = {f.name for f in fields(ControllerConfig)}
_TRAIN_KEYS = None  # Lazy init after TrainConfig import


# --- 3. Save / Load ---
def save_model(path, *, config, grad_net=None, extra=None, history=None):
    """Save model checkpoint (weights + history + metadata)."""
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
    }
    history_to_save = None
    if history is not None:
        history_to_save = {
            "iters": list(history["iters"]),
            "loss": list(history["loss"]),
        }
        if "phase" in history:
            history_to_save["phase"] = list(history["phase"])
        if "val_mse" in history:
            history_to_save["val_mse"] = list(history["val_mse"])
    ckpt["history"] = history_to_save
    torch.save(ckpt, path)


def load_gradnet(path, *, config, device="cpu"):
    """Load model checkpoint. Returns (grad_net, meta, history)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if ckpt.get("grad_net") is None:
        raise ValueError(f"No grad_net weights in checkpoint: {path}")

    sd = ckpt["grad_net"]
    sd = {k: v for k, v in sd.items() if not k.startswith("_physics.")}

    grad_net = GradNet(config).to(device)
    grad_net.load_state_dict(sd, strict=True)
    grad_net.eval()

    hist = ckpt.get("history")
    if hist is not None:
        hist_orig = hist
        hist = {"iters": np.asarray(hist_orig["iters"]), "loss": np.asarray(hist_orig["loss"])}
        if "phase" in hist_orig:
            hist["phase"] = np.asarray(hist_orig["phase"])
        if "val_mse" in hist_orig:
            hist["val_mse"] = np.asarray(hist_orig["val_mse"])
    return grad_net, ckpt.get("meta", {}), hist


def _ckpt_path(ckpt_dir, cfg, config, train_overrides=None):
    """Build deterministic checkpoint filename from controller + problem config."""
    base = (
        f"{cfg.kind}_useLQR{int(cfg.use_lqr)}_{cfg.train_mode}"
        f"_pre{int(cfg.pretrain)}_adapt{int(cfg.adaptive)}"
        f"_sup{int(cfg.supervision or False)}"
        f"_d{config.n_states}_m{config.n_controls}_seed{config.seed}"
    )
    if train_overrides and "n_candidates" in train_overrides:
        base += f"_nc{train_overrides['n_candidates']}"
    return os.path.join(ckpt_dir, base + ".pt")


# --- 4. Training Entry Point ---
def train_controllers_from_config(
    controller_configs=None,
    config=None,
    train_cfg=None,
    data=None,
    val_data=None,
    device="cpu",
    verbose=True,
    force_retrain=False,
):
    """
    Trains one or multiple controllers based on the provided configuration.

    Models are saved after training and loaded from cache on subsequent runs
    (unless force_retrain=True).

    Parameters
    ----------
    controller_configs : dict, optional
        Dictionary where keys are model names and values are config dicts.
        If None, uses all DEFAULT_CONTROLLER_CONFIGS.
        Partial updates are allowed (e.g., passing just {"GradNet": {"enabled": True}}).
    force_retrain : bool, default False
        If True, retrain even if a checkpoint exists.
    """
    from problems import get_results_dir
    from controls.train import TrainConfig

    global _TRAIN_KEYS
    if _TRAIN_KEYS is None:
        _TRAIN_KEYS = {f.name for f in fields(TrainConfig)}

    defaults = get_default_controller_configs()
    train_overrides_by_name = {}  # Per-controller TrainConfig overrides (e.g. n_candidates)

    if controller_configs is None:
        final_configs = defaults
    else:
        for name, override_cfg in controller_configs.items():
            if not isinstance(override_cfg, dict):
                if name not in defaults:
                    defaults[name] = override_cfg
                continue
            # Split into controller vs train overrides
            ctrl_overrides = {k: v for k, v in override_cfg.items() if k in _CONTROLLER_KEYS}
            train_overrides = {k: v for k, v in override_cfg.items() if k in _TRAIN_KEYS}
            if train_overrides:
                train_overrides_by_name[name] = train_overrides

            if name in defaults:
                for k, v in ctrl_overrides.items():
                    setattr(defaults[name], k, v)
            else:
                defaults[name] = ControllerConfig(**ctrl_overrides)
        final_configs = defaults

    active_configs = {
        name: cfg for name, cfg in final_configs.items()
        if cfg.enabled
    }

    if not active_configs:
        if verbose: print("No controllers enabled. Exiting.")
        return {}, {}, {}

    ckpt_dir = get_results_dir(config, "saved_models")
    os.makedirs(ckpt_dir, exist_ok=True)

    trained_models = {}
    trained_controllers = {}
    histories = {}

    if verbose:
        print(f"Training {len(active_configs)} models on {device}...")

    for name, cfg in active_configs.items():
        path = _ckpt_path(ckpt_dir, cfg, config, train_overrides=train_overrides_by_name.get(name))

        # Try loading from cache
        if (not force_retrain) and os.path.exists(path):
            if verbose:
                print(f"\n--- Loading {name} from cache ---")
            model, meta, history = load_gradnet(path, config=config, device=device)
            trained_models[name] = model
            trained_controllers[name] = Control(config, grad_net=model)
            histories[name] = history
            continue

        if verbose:
            print(f"\n--- Training {name} ---")
            pre_str = " + pretrain" if cfg.pretrain else ""
            print(f"Mode: {cfg.train_mode}{pre_str} | LQR: {cfg.use_lqr}")

        model = GradNet(config, use_lqr=cfg.use_lqr).to(device)

        from controls.train import train

        # Merge per-controller train overrides (e.g. n_candidates) into base train_cfg
        effective_train_cfg = train_cfg
        if name in train_overrides_by_name:
            effective_train_cfg = replace(train_cfg, **train_overrides_by_name[name])

        history = train(
            model,
            config=config,
            train_config=effective_train_cfg,
            data=data,
            val_data=val_data,
            mode=cfg.train_mode,
            pretrain=cfg.pretrain,
            supervision=cfg.supervision,
            adaptive=cfg.adaptive,
        )

        # Save checkpoint
        meta_out = {
            "kind": cfg.kind, "use_lqr": cfg.use_lqr, "train_mode": cfg.train_mode,
            "pretrain": cfg.pretrain, "adaptive": cfg.adaptive,
            "supervision": cfg.supervision,
        }
        save_model(path, config=config, grad_net=model, extra=meta_out, history=history)
        if verbose:
            print(f"  Saved to {path}")

        trained_models[name] = model
        trained_controllers[name] = Control(config, grad_net=model)
        histories[name] = history

    return trained_models, trained_controllers, histories

# Alias used by notebooks
train_controllers = train_controllers_from_config
