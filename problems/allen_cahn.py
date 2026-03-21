import numpy as np

from .base import (
    BaseOCP,
    _build_neumann_operators,
    attach_ocp,
    spatial_control_matrix,
    sample_initial_conditions_fourier,
)
from controls.lqr import LQR


def get_default_config():
    """Allen–Cahn system-specific default config (n_states, n_controls, time, nu, solver, etc.)."""
    return {
        "n_states": 32,
        "n_controls": 1,
        "t1_initial": 10,
        "t1_scale": 6 / 5,
        "t1_max": 60.0,
        "nu": 0.1,
        "control_width": 0.4,
        "ic_modes": 9,  
        "ic_scale": 1.0,
        "ic_use_sine": False,
        "ic_use_cosine": True,
        "ocp_solver": "indirect",
        "direct_n_init_nodes": 50,
        "indirect_tol": 1e-03,
        "indirect_max_nodes": 1500,
        "fp_tol": 0.01,
        "conv_tol": 0.01,
    }


def attach_config(config, overrides=None):
    """Attach Allen-Cahn OCP and system-specific attributes to config."""
    attach_ocp(config, AllenCahnOCP, get_default_config(), overrides)


class AllenCahnOCP(BaseOCP):
    """Allen–Cahn optimal control problem (Neumann BC discretization)."""

    def __init__(self, config):
        self._config = config
        n_states = int(config.n_states)
        n_controls = int(config.n_controls)

        # physics params (nu can be overridden for model-mismatch evaluation)
        self.nu = config.nu
        self.R = 0.5

        # operators + quadrature weights (Neumann BC)
        self.xi, self.D, self.D1, self.D2, self.w_flat, self.w = _build_neumann_operators(n_states)

        width = getattr(config, "control_width", 0.2)
        B = spatial_control_matrix(
            self.xi, n_controls, width, domain_type="symmetric", x_min=-1.0, x_max=1.0
        )




        # linearization point - target stable steady state (-1 or 1)
        # Using -1 as default (both -1 and 1 are stable)
        target_value = 0.0 # -1.0 or 1.0
        X_bar = np.full((n_states, 1), target_value)
        U_bar = np.zeros((n_controls, 1))

        # linearized dynamics at x = X_bar (stable steady state)
        # d/dx (x - x^3) = 1 - 3x^2
        # At x = -1: 1 - 3(-1)^2 = 1 - 3 = -2
        # At x = 1:  1 - 3(1)^2 = 1 - 3 = -2
        # So A = nu*D2 + diag(-2) = nu*D2 - 2*I (stable!)
        diag_term = 1.0 - 3.0 * (target_value ** 2)  # = -2 for ±1
        #A = self.nu * self.D2 + diag_term * np.eye(n_states) + np.diag(self.alpha_flat)
        A = self.nu * self.D2 + diag_term * np.eye(n_states) 

        # cost matrices (state weight uses quadrature weights)
        Q = np.diag(self.w_flat)
        Rm = np.diag([self.R] * n_controls)

        self.X_bar = X_bar
        self.U_bar = U_bar

        self.n_states = n_states
        self.n_controls = n_controls

        self._A = A
        self._B = B
        self._Q = Q
        self._R = Rm

        self.LQR = LQR(X_bar, U_bar, self._A, self._B, self._Q, self._R, P=None)
        self.B = self._B
        # optimal control map used by BaseOCP.U_star: U* = RBT @ dVdX
        self.RBT = -self._B.T / (2.0 * self.R)   # shape (m,n)


    def dynamics(self, X, U):
        """
        Allen–Cahn:
          dXdt = nu * D2 * X + X - X^3 
        
        Forcing (Burgers-style): X * alpha * exp(-gamma*X), alpha = bump in |xi|<=1/5.
        """
        flat_out = X.ndim < 2
        X = X.reshape(self.n_states, -1)
        U = U.reshape(self.n_controls, -1)

        dXdt = (self.nu * (self.D2 @ X)) + X - (X ** 3) + (self.B @ U)

        return dXdt.flatten() if flat_out else dXdt

    def jacobians(self, X, U, F0=None):
        """
        Returns:
          dFdX: (n,n,N)
          dFdU: (n,m,N)
        """
        X = X.reshape(self.n_states, -1)
        N = X.shape[1]

        # d/dx (x - x^3) = 1 - 3x^2
        diag_term = 1.0 - 3.0 * (X ** 2)  # (n,N)

        dFdX = np.empty((self.n_states, self.n_states, N), dtype=float)
        base = self.nu * self.D2
        idx = np.arange(self.n_states)
 

        for k in range(N):
            J = base.copy()
            J[idx, idx] += diag_term[:, k]

            dFdX[:, :, k] = J

        dFdU = np.tile(self.B[:, :, None], (1, 1, N))
        return dFdX, dFdU

    def bvp_dynamics(self, t, X_aug):
        """
        Augmented PMP dynamics for solve_bvp:
        X_aug = [X; A; V] with A = dVdX (costate), V is accumulated cost.
        Returns d/dt [X; A; V] = [ dXdt; dAdt; -L ].
        """
        X = X_aug[:self.n_states].reshape(self.n_states, -1)                 # (n,N)
        A = X_aug[self.n_states:2*self.n_states].reshape(self.n_states, -1)  # (n,N)

        # optimal control from costate
        U = self.U_star(X, A)                                                # (m,N)

        # dynamics: dXdt = nu D2 X + X - X^3 + X*alpha*exp(-gamma*X) + B U
        dXdt = (self.nu * (self.D2 @ X)) + X - (X ** 3)  + (self.B @ U)

        X_err = X - self.X_bar
        wX_err = self.w * X_err
        dLdX = 2.0 * wX_err
        dAdt = -dLdX - (self.nu * (self.D2.T @ A)) - ((1.0 - 3.0 * (X ** 2)) * A)


        # running cost (1,N) - pass wX_err for consistency
        L = np.atleast_2d(self.running_cost(X, U, wX_err))

        return np.vstack((dXdt, dAdt, -L))

    def sample_initial_conditions(self, n: int, seed=None, **kwargs):
        """Allen–Cahn: Fourier-cosine sum (unified)."""
        cfg = self._config
        modes = kwargs.get("K") or kwargs.get("modes") or getattr(cfg, "ic_modes", 5)
        scale = kwargs.get("scale") or getattr(cfg, "ic_scale", 1.0)
        use_sine = kwargs.get("use_sine") if "use_sine" in kwargs else getattr(cfg, "ic_use_sine", False)
        use_cosine = kwargs.get("use_cosine") if "use_cosine" in kwargs else getattr(cfg, "ic_use_cosine", True)
        return sample_initial_conditions_fourier(
            self.xi, self.n_states, n, modes=modes, scale=scale,
            use_sine=use_sine, use_cosine=use_cosine, domain_type="symmetric",
            seed=seed, **{k: v for k, v in kwargs.items() if k not in ("K", "modes", "scale", "use_sine", "use_cosine")}
        )