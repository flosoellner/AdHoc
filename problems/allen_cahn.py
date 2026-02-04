import numpy as np

from .base import BaseOCP, _build_neumann_operators, _default_control_matrix
from controls.lqr import LQR


def get_default_config():
    """Allen–Cahn system-specific default config (n_states, n_controls, time, nu, solver, etc.)."""
    return {
        "n_states": 48,
        "n_controls": 2,
        "t1_initial": 6.0,
        "t1_scale": 6 / 5,
        "t1_max": 30.0,
        "nu": 0.15,
        "ocp_solver": "indirect",
        "direct_n_init_nodes": 50,
        "indirect_tol": 1e-03,
        "indirect_max_nodes": 1500,
    }


def attach_config(config, overrides=None):
    """Attach Allen–Cahn OCP and system-specific attributes to config."""
    overrides = overrides or {}
    defaults = get_default_config()
    for k, v in defaults.items():
        setattr(config, k, overrides.get(k, v))
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
        dFdX, _ = cfg.ocp.jacobians(x, np.zeros((cfg.n_controls,)))
        return dFdX

    config.f_controlled = f_controlled
    config.jacobian_f = jacobian_f


class AllenCahnOCP(BaseOCP):
    """Allen–Cahn optimal control problem (Neumann BC discretization)."""

    def __init__(self, config):
        n_states = int(config.n_states)
        n_controls = int(config.n_controls)

        # physics params (nu can be overridden for model-mismatch evaluation)
        self.nu = getattr(config, 'nu', 0.15)
        self.R = 0.5

        # operators + quadrature weights (Neumann BC)
        self.xi, self.D, self.D1, self.D2, self.w_flat, self.w = _build_neumann_operators(n_states)

        # Control matrix B: Burgers-style (2 regions) when n_controls==2, else default
        kappa = 25.0
        if n_controls == 2:
            B = np.hstack((
                (-4/5 <= self.xi) & (self.xi <= -2/5),
                (2/5 <= self.xi) & (self.xi <= 4/5)
            ))
            B = -kappa * B * np.hstack((
                (self.xi + 4/5) * (self.xi + 2/5),
                (self.xi - 2/5) * (self.xi - 4/5)
            ))
            B = np.abs(B)
        else:
            B = _default_control_matrix(n_states, n_controls)

        # Forcing term (Burgers-style): X * alpha * exp(-gamma*X), alpha = bump in |xi|<=1/5
        self.gamma = 0.1
        self.alpha = np.abs(self.xi) <= 1/5
        self.alpha = -kappa * self.alpha * (self.xi + 1/5) * (self.xi - 1/5)
        self.alpha = np.abs(self.alpha)
        self.alpha_flat = self.alpha.flatten()

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
        A = self.nu * self.D2 + diag_term * np.eye(n_states) + np.diag(self.alpha_flat)

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
          dXdt = nu * D2 * X + X - X^3 + X * alpha * exp(-gamma*X) + B * U
        
        Forcing (Burgers-style): X * alpha * exp(-gamma*X), alpha = bump in |xi|<=1/5.
        """
        flat_out = X.ndim < 2
        X = X.reshape(self.n_states, -1)
        U = U.reshape(self.n_controls, -1)

        dXdt = (self.nu * (self.D2 @ X)) + X - (X ** 3) + (self.B @ U)
        dXdt += X * self.alpha * np.exp(-self.gamma * X)
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
        # Forcing Jacobian: d/dX [ X*alpha*exp(-gamma*X) ] is diag( alpha*exp(-gamma*X)*(1 - gamma*X) )
        forcing_diag = self.alpha * np.exp(-self.gamma * X) * (1.0 - self.gamma * X)  # (n,N)

        for k in range(N):
            J = base.copy()
            J[idx, idx] += diag_term[:, k]
            J[idx, idx] += forcing_diag[:, k]
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
        aeX = self.alpha * np.exp(-self.gamma * X)
        dXdt = (self.nu * (self.D2 @ X)) + X - (X ** 3) + X * aeX + (self.B @ U)

        # costate dynamics: dA/dt = -∂H/∂X = -∂L/∂X - (∂f/∂X)^T A
        X_err = X - self.X_bar
        wX_err = self.w * X_err
        dLdX = 2.0 * wX_err
        dAdt = -dLdX - (self.nu * (self.D2.T @ A)) - ((1.0 - 3.0 * (X ** 2)) * A)
        dAdt -= aeX * (1.0 - self.gamma * X) * A

        # running cost (1,N) - pass wX_err for consistency
        L = np.atleast_2d(self.running_cost(X, U, wX_err))

        return np.vstack((dXdt, dAdt, -L))

    def sample_initial_conditions(self, n: int, seed=None, K: int = 10):
        """Allen–Cahn: Fourier-cosine sum on grid (single-IC formula)."""
        if seed is not None:
            np.random.seed(seed)
        d = self.n_states
        xi = self.xi.flatten()
        X0 = np.zeros((d, n))
        for k in range(1, K + 1):
            ak = (2.0 * np.random.rand(1, n) - 1.0) / float(k)
            X0 += ak * np.cos(k * np.pi * xi).reshape(d, 1)
        return X0