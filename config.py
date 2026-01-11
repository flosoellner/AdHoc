

# Problem dimensions
n_states = 32
n_controls = 3

# Time horizon parameters
t1_initial = 30.0
t1_scale = 6/5
t1_max = 150.0

system = 'allen_cahn'






# Seed
seed = 4


def create_config(**overrides):
    """
    Create the central configuration object (attribute access everywhere).
    
    Parameters
    ----------
    **overrides : dict
        Any parameters to override defaults
        
    Returns
    -------
    config : types.SimpleNamespace
        Configuration object (supports `config.seed`, `config.n_states`, ...)
    """
    from types import SimpleNamespace
    import numpy as np

    config = SimpleNamespace(
        seed=seed,
        n_states=n_states,
        n_controls=n_controls,
        t1_initial=t1_initial,
        t1_scale=t1_scale,
        t1_max=t1_max,
        system=system,
        fp_tol=5e-03
    )

    # Apply overrides (e.g. create_config(T_initial=30.0))
    for k, v in overrides.items():
        setattr(config, k, v)

    # Attach problem-specific pieces
    if config.system == "burgers":
        # Local import to avoid circular imports
        from problems.burgers import BurgersOCP

        config.ocp_solver = "indirect"
        config.direct_n_init_nodes = 50
        config.indirect_tol = 1e-05
        config.indirect_max_nodes=1500

        config.ocp = BurgersOCP(config)
        config.B = config.ocp.B
        config.q = config.ocp.R
        config.x_f = config.ocp.X_bar.flatten()


        config.xi = config.ocp.xi
        config.w = config.ocp.w
        config.norm = config.ocp.norm

        # Expose OCP methods directly on config
        config.dynamics = config.ocp.dynamics
        config.running_cost = config.ocp.running_cost
        config.running_cost_gradient = config.ocp.running_cost_gradient

        # Wrapper for f_controlled signature
        def f_controlled(t, x, u, cfg):
            return cfg.ocp.dynamics(x, u)

        # Jacobian function (dF/dX at x)
        def jacobian_f(x, cfg):
            dFdX, _dFdU = cfg.ocp.jacobians(x, np.zeros((cfg.m,)))
            return dFdX

        config.f_controlled = f_controlled
        config.jacobian_f = jacobian_f
    elif config.system == "allen_cahn":
        from problems.allen_cahn import AllenCahnOCP

        # keep these consistent with burgers branch (solver params etc if needed)
        config.ocp_solver = "indirect"
        config.direct_n_init_nodes = 50
        config.indirect_tol = 1e-05
        config.indirect_max_nodes = 1500

        config.ocp = AllenCahnOCP(config)
        config.B = config.ocp.B
        config.q = config.ocp.R
        config.x_f = config.ocp.X_bar.flatten()

        config.xi = config.ocp.xi
        config.w = config.ocp.w
        config.norm = config.ocp.norm

        config.dynamics = config.ocp.dynamics
        config.running_cost = config.ocp.running_cost
        config.running_cost_gradient = config.ocp.running_cost_gradient

        def f_controlled(t, x, u, cfg):
            return cfg.ocp.dynamics(x, u)

        def jacobian_f(x, cfg):
            dFdX, _dFdU = cfg.ocp.jacobians(x, np.zeros((cfg.n_controls,)))
            return dFdX

        config.f_controlled = f_controlled
        config.jacobian_f = jacobian_f

    else:
        raise ValueError(f"Unknown system: {config.system}")

    return config
