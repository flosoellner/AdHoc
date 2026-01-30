import numpy as np

from .base import BaseOCP, _build_dirichlet_operators
from controls.lqr import LQR



def get_default_config():
    """KdV system-specific default config."""
    return {
        "n_states": 32,      # KdV often needs higher resolution for solitons
        "n_controls": 2,
        "t1_initial": 5.0,
        "t1_scale": 1.1,
        "t1_max": 15.0,
        "ocp_solver": "indirect",
        "direct_n_init_nodes": 50,
        "indirect_tol": 5e-3,
        "indirect_max_nodes": 4000,
    }

def attach_config(config, overrides=None):
    """Attach KdV OCP and system-specific attributes to config."""
    overrides = overrides or {}
    defaults = get_default_config()
    for k, v in defaults.items():
        setattr(config, k, overrides.get(k, v))
        
    config.ocp = kdvOCP(config)
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
        # Linearized around current state x
        dFdX, _ = cfg.ocp.jacobians(x, np.zeros((cfg.n_controls,)))
        return dFdX

    config.f_controlled = f_controlled
    config.jacobian_f = jacobian_f

class kdvOCP(BaseOCP):
    def __init__(self, config):
        n_states = config.n_states
        n_controls = config.n_controls
        
        # Physics 
        self.delta = 0.1
        self.nu = 0.0     
        self.eta = 0.1     
        self.R = 0.1       

        # Call the updated base builder
        # Note: Added D3 to the return values
        self.xi, self.D, self.D2, self.D3, self.w_flat, self.w = _build_dirichlet_operators(n_states)

        # Actuators (Simplified)
        self.B = np.zeros((n_states, n_controls))
        self.B[:n_states//2, 0] = 1.0
        self.B[n_states//2:, 1] = 1.0
        self.RBT = - self.B.T / (2. * self.R)

        # LQR
        X_bar = np.zeros((n_states, 1))
        U_bar = np.zeros((n_controls, 1))
        A_lin = -(self.delta**2) * self.D3 + self.nu * self.D2
        
        self.X_bar, self.U_bar = X_bar, U_bar
        self.n_states, self.n_controls = n_states, n_controls
        self.LQR = LQR(X_bar, U_bar, A_lin, self.B, np.diag(self.w_flat), np.diag([self.R]*n_controls))

    def dynamics(self, X, U):
        """
        KdV Dynamics: dXdt = -eta * X * (D @ X) - delta^2 * (D3 @ X) + B @ U
        """
        flat_out = X.ndim < 2
        X = X.reshape(self.n_states, -1)
        U = U.reshape(self.n_controls, -1)

        # Standard dispersive + nonlinear terms
        dXdt = (
            - self.eta * X * (self.D @ X) 
            - (self.delta**2) * (self.D3 @ X) 
            + self.nu * (self.D2 @ X)        # <--- New Diffusion Term
            + self.B @ U
        )
        return dXdt.flatten() if flat_out else dXdt

    def jacobians(self, X, U, F0=None):
        X = X.reshape(self.n_states, -1)
        # d/dx (-eta * x * Dx) = -eta * (diag(Dx) + diag(x)@D)
        Dx = self.D @ X

        # Linear part: -delta^2 * D3 + nu * D2 (must match dynamics)
        base_jac = -(self.delta**2) * self.D3 + self.nu * self.D2

        dFdX = np.zeros((self.n_states, self.n_states, X.shape[1]))
        for k in range(X.shape[1]):
            dFdX[:, :, k] = base_jac - self.eta * (np.diag(Dx[:, k]) + np.diag(X[:, k]) @ self.D)

        dFdU = np.tile(np.expand_dims(self.B, -1), (1, 1, X.shape[-1]))
        return dFdX, dFdU

    def bvp_dynamics(self, t, X_aug):
        X = X_aug[:self.n_states].reshape(self.n_states, -1)
        A = X_aug[self.n_states:2*self.n_states].reshape(self.n_states, -1)

        U = self.U_star(X, A)
        wX = self.w * X

        # 1. State dynamics must match dynamics(): include diffusion
        dXdt = (
            - self.eta * X * (self.D @ X)
            - (self.delta**2) * (self.D3 @ X)
            + self.nu * (self.D2 @ X)
            + self.B @ U
        )

        # 2. Costate: dAdt = -dH/dX. Adjoint of -delta^2*D3 is +delta^2*D3'; of -eta*x*(Dx) is eta*((Dx)*lambda + D'@(x*lambda))
        dAdt = (
            - 2.0 * wX
            + self.eta * ((self.D @ X) * A + self.D.T @ (X * A))
            + (self.delta**2) * (self.D3.T @ A)
            - self.nu * (self.D2.T @ A)
        )

        L = np.atleast_2d(self.running_cost(X, U, wX))
        return np.vstack((dXdt, dAdt, -L))
        


    def sample_initial_conditions(self, n: int, seed=None, **kwargs):
        if seed is not None: np.random.seed(seed)
        xi = self.xi.flatten()
        X0 = np.zeros((self.n_states, n))
        
        for i in range(n):
            # A tiny, wide hump in the middle
            amp = 0.2  # Very low amplitude (easy for solver)
            width = 0.2 # Wide (easy for grid to resolve)
            center = 0.0 
            X0[:, i] = 0.55 * amp * np.exp(-((xi - center)**2) / (2 * width**2))
            
        return X0