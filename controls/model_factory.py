import os
import numpy as np
import torch
from types import SimpleNamespace
from controls.nn import GradNet

DEFAULT_CONTROLLER_CONFIGS = {

    "GradNet": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": False,
        "train_mode": "unsupervised",
        "adaptive": False,
        "supervision": None,
    },
    "GradNet (sup)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": False,
        "train_mode": "unsupervised",
        "adaptive": False,
        "supervision": True,
    },

    "GradQRNet": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": True,
        "train_mode": "unsupervised",
        "adaptive": False,
        "supervision": None,
    },
    "GradQRNet (sup)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": True,
        "train_mode": "unsupervised",
        "adaptive": False,
        "supervision": True,
    },

    "GradNet (pre)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": False,
        "train_mode": "hybrid",
        "adaptive": False,
        "supervision": None,
    },
    "GradNet (pre/sup)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": False,
        "train_mode": "hybrid",
        "adaptive": False,
        "supervision": True,
    },

    "GradQRNet (pre)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": True,
        "train_mode": "hybrid",
        "adaptive": False,
        "supervision": None,
    },
    "GradQRNet (pre/sup)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": True,
        "train_mode": "hybrid",
        "adaptive": False,
        "supervision": True,
    },

    "GradNet (ad)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": False,
        "train_mode": "unsupervised",
        "adaptive": True,
        "supervision": None,
    },
    "GradNet (sup/ad)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": False,
        "train_mode": "unsupervised",
        "adaptive": True,
        "supervision": True,
    },

    "GradQRNet (ad)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": True,
        "train_mode": "unsupervised",
        "adaptive": True,
        "supervision": None,
    },
    "GradQRNet (sup/ad)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": True,
        "train_mode": "unsupervised",
        "adaptive": True,
        "supervision": True,
    },

    "GradNet (pre/ad)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": False,
        "train_mode": "hybrid",
        "adaptive": True,
        "supervision": None,
    },
    "GradNet (pre/sup/ad)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": False,
        "train_mode": "hybrid",
        "adaptive": True,
        "supervision": True,
    },

    "GradQRNet (pre/ad)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": True,
        "train_mode": "hybrid",
        "adaptive": True,
        "supervision": None,
    },
    "GradQRNet (sup/ad)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": True,
        "train_mode": "unsupervised",
        "adaptive": True,
        "supervision": True,
    },
    "GradQRNet (pre/sup/ad)": {
        "enabled": False,
        "kind": "gradnet",
        "use_lqr": True,
        "train_mode": "hybrid",
        "adaptive": True,
        "supervision": True,
    },

}

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
    # Save phase information if present (for hybrid models)
    if history is not None and "phase" in history:
        history_to_save["phase"] = list(history["phase"])
    # Save val_mse if present
    if history is not None and "val_mse" in history:
        history_to_save["val_mse"] = list(history["val_mse"])
    ckpt["history"] = history_to_save
    torch.save(ckpt, path)


def load_gradnet(path, *, config, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if ckpt.get("grad_net") is None:
        raise ValueError(f"No grad_net weights in checkpoint: {path}")

    sd = ckpt["grad_net"]
    sd = {k: v for k, v in sd.items() if not k.startswith("_physics.")}

    has_sequential_style = any(k.startswith("net.0.") for k in sd.keys())



    grad_net = GradNet(config).to(device)

    grad_net.load_state_dict(sd, strict=True)
    grad_net.eval()

    hist = ckpt.get("history")
    if hist is not None:
        hist_orig = hist  # Keep reference to original
        hist = {"iters": np.asarray(hist["iters"]), "loss": np.asarray(hist["loss"])}
        # Restore phase information if present (check original dict)
        if "phase" in hist_orig:
            hist["phase"] = np.asarray(hist_orig["phase"])
        # Restore val_mse if present (check original dict)
        if "val_mse" in hist_orig:
            hist["val_mse"] = np.asarray(hist_orig["val_mse"])
    return grad_net, ckpt.get("meta", {}), hist


def train_or_load_gradnet(
    *,
    config,
    kind: str,
    use_lqr: bool,
    train_mode: str,            # "supervised" | "unsupervised" | "hybrid"
    adaptive: bool = False, 
    supervision: bool | None = False,
    train_loader=None,
    val_loader=None,  # NEW: validation loader
    train_cfg,
    ckpt_dir: str = None,
    force_retrain: bool = False,
    device: str = "cpu",
    data=None,
    val_data=None,  # NEW: validation data dict
):
    from controls.train import train_loop, loss_unified, make_loader_XG
    from problems import get_results_dir
    
    if supervision is None:
        supervision = False
    
    if ckpt_dir is None:
        ckpt_dir = get_results_dir(config, "saved_models")
    
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

    # Create validation loader if validation data is provided
    val_loader_actual = val_loader
    if val_loader_actual is None and val_data is not None:
        val_loader_actual = make_loader_XG(val_data, train_cfg, shuffle=False)

    sup_loader = None
    if supervision:
        if data is None:
            raise ValueError("supervision=True requires `data` (X,dVdX)")
        sup_loader = make_loader_XG(data, train_cfg, shuffle=True)

    if train_mode == "supervised":
        if (train_loader is None) and (data is None):
            raise ValueError("supervised requires either `data` or `train_loader`")
        grad_net, history = train_loop(
            grad_net,
            train_loader if train_loader is not None else make_loader_XG(data, train_cfg),
            loss_unified,
            sup_cfg,
            mode="supervised",
            val_loader=val_loader_actual,  # NEW
        )

    elif train_mode == "unsupervised":
        # Calculate expected total iterations (same as hybrid would have)
        # This ensures unsupervised-only plots end at the same point as hybrid
        if data is not None:
            sup_loader_temp = make_loader_XG(data, train_cfg, shuffle=True)
            expected_sup_iters = train_cfg.sup_epochs * len(sup_loader_temp)
        else:
            # Estimate: if no data provided, use a default or estimate
            # This shouldn't happen in normal usage, but handle gracefully
            expected_sup_iters = 0
        
        expected_unsup_iters = train_cfg.unsup_epochs * train_cfg.unsup_n_steps
        expected_total_iters = expected_sup_iters + expected_unsup_iters
        
        # unsupervised call:
        grad_net, history = train_loop(
            grad_net, None, loss_unified, unsup_cfg,
            mode="unsupervised", config=config, adaptive=adaptive,
            supervision=supervision, sup_loader=sup_loader, sup_every=1,
            val_loader=val_loader_actual,  # NEW
        )
        
        # Offset so unsupervised-only ends at expected_total_iters (same as hybrid)
        # Use actual iteration count in case it differs from expected
        actual_unsup_iters = len(history["iters"])
        # Calculate offset: unsupervised should end where hybrid would end
        # If actual matches expected: offset = expected_sup_iters (correct!)
        # If actual differs: still align end points correctly
        start_offset = expected_total_iters - actual_unsup_iters
        # Apply offset: now unsupervised-only iters start after sup_iters would end
        history["iters"] = history["iters"] + start_offset


    elif train_mode == "hybrid":
        if data is None:
            raise ValueError("hybrid requires data")

        grad_net, hist_sup = train_loop(
            grad_net,
            make_loader_XG(data, train_cfg),
            loss_unified,
            sup_cfg,
            mode="supervised",
            supervision=supervision,  # NEW: pass supervision parameter
            val_loader=val_loader_actual,
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
            val_loader=val_loader_actual,
        )

        # ... rest of existing code ...


        # Combine supervised and unsupervised histories
        it_sup, lo_sup = hist_sup["iters"], hist_sup["loss"]
        it_uns, lo_uns = hist_unsup["iters"], hist_unsup["loss"]
        offset = int(it_sup[-1]) if len(it_sup) else 0

        history = {
            "iters": np.concatenate([it_sup, it_uns + offset]),
            "loss":  np.concatenate([lo_sup, lo_uns]),
            "phase": np.array((["sup"] * len(it_sup)) + (["unsup"] * len(it_uns))),
        }
        
        # Add validation MSE if available
        if "val_mse" in hist_sup:
            val_mse_sup = hist_sup["val_mse"]
            val_mse_unsup = hist_unsup.get("val_mse", [])
            if len(val_mse_unsup) > 0:
                history["val_mse"] = np.concatenate([val_mse_sup, val_mse_unsup])
            else:
                history["val_mse"] = val_mse_sup

    else:
        raise ValueError("train_mode must be 'supervised', 'unsupervised', or 'hybrid'")

    meta_out = {"kind": kind, "use_lqr": use_lqr, "train_mode": train_mode, "adaptive": adaptive, "supervision": supervision}
    save_model(ckpt_path, config=config, grad_net=grad_net, value_net=None, extra=meta_out, history=history)

    return (grad_net, meta_out, history)




def train_controllers_from_config(
    controller_configs,
    *,
    config,
    train_cfg,
    data=None,
    val_data=None,  # NEW: validation data
    device="cpu",
    verbose=True,
):
    """
    Train multiple controllers from a configuration dictionary.
    
    Parameters
    ----------
    controller_configs : dict
        Dictionary mapping controller names to their configurations.
        Each config should have:
        - "enabled": bool - whether to train this controller
        - "kind": str - model kind ("gradnet", "resnet", etc.)
        - "use_lqr": bool - whether to use LQR baseline
        - "train_mode": str - "supervised", "unsupervised", or "hybrid"
        - "adaptive": bool - whether to use adaptive batch sizing
        - "supervision": bool - whether to use supervision in unsupervised mode
    config : SimpleNamespace
        Problem configuration
    train_cfg : TrainConfig
        Training configuration
    data : dict, optional
        Training data (required for supervised/hybrid modes)
    val_data : dict, optional
        Validation data for computing validation MSE
    device : str, default "cpu"
        Device to train on
    verbose : bool, default True
        Whether to print training progress
        
    Returns
    -------
    trained_models : dict
        Dictionary mapping controller names to trained models
    trained_controllers : dict
        Dictionary mapping controller names to Control objects
    histories : dict
        Dictionary mapping controller names to training histories
    """
    from controls.nn import Control
    
    trained_models = {}
    trained_controllers = {}
    histories = {}
    
    for name, cfg in controller_configs.items():
        if not cfg.get("enabled", False):
            continue
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training: {name}")
            print(f"{'='*60}")
        
        model, meta, hist = train_or_load_gradnet(
            config=config,
            kind=cfg["kind"],
            use_lqr=cfg["use_lqr"],
            train_mode=cfg["train_mode"],
            adaptive=cfg["adaptive"],
            supervision=cfg["supervision"],
            train_cfg=train_cfg,
            data=data,
            val_data=val_data,  # NEW: pass validation data
            device=device,
        )
        
        ctrl = Control(config, grad_net=model)
        
        trained_models[name] = model
        trained_controllers[name] = ctrl
        histories[name] = hist
    
    return trained_models, trained_controllers, histories

def get_default_controller_configs():
    """Get a copy of the default controller configurations."""
    import copy
    return copy.deepcopy(DEFAULT_CONTROLLER_CONFIGS)

def train_controllers(
    *,
    config,
    train_cfg,
    data=None,
    val_data=None,  # NEW: validation data
    controller_configs=None,
    device="cpu",
    verbose=True,
):
    """
    Train controllers using default configs or provided overrides.
    
    Parameters
    ----------
    config : SimpleNamespace
        Problem configuration
    train_cfg : TrainConfig
        Training configuration
    data : dict, optional
        Training data (required for supervised/hybrid modes)
    val_data : dict, optional
        Validation data for computing validation MSE
    controller_configs : dict, optional
        Controller configurations. If None, uses defaults.
        Can be a partial dict to override only specific controllers.
        Use {"Model 3 (Hybrid)": {"enabled": True}} to enable just one.
    device : str, default "cpu"
        Device to train on
    verbose : bool, default True
        Whether to print training progress
        
    Returns
    -------
    trained_models : dict
        Dictionary mapping controller names to trained models
    trained_controllers : dict
        Dictionary mapping controller names to Control objects
    histories : dict
        Dictionary mapping controller names to training histories
    """
    if controller_configs is None:
        controller_configs = get_default_controller_configs()
    else:
        # Merge with defaults, allowing partial overrides
        defaults = get_default_controller_configs()
        for name, override_cfg in controller_configs.items():
            if name in defaults:
                defaults[name].update(override_cfg)
            else:
                defaults[name] = override_cfg
        controller_configs = defaults
    
    return train_controllers_from_config(
        controller_configs,
        config=config,
        train_cfg=train_cfg,
        data=data,
        val_data=val_data,  # NEW: pass validation data
        device=device,
        verbose=verbose,
    )

