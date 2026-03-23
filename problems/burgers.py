import numpy as np

from .base import (
    BaseOCP,
    _build_dirichlet_operators,
    attach_ocp,
    spatial_control_matrix,
    sample_initial_conditions_fourier,
)
from controls.lqr import LQR


def get_default_config():
    """Burgers system-specific default config (n_states, n_controls, time, solver, nu, etc.)."""
    return {
        "n_states": 32,
        "n_controls": 2,
        "t1_initial": 20.0,
        "t1_scale": 6 / 5,
        "t1_max": 60.0,
        "nu": 0.012,
        "control_width": 0.2,
        "ic_modes": 6,
        "ic_scale": 1.0,
        "ic_use_sine": True,
        "ic_use_cosine": False,
        "ocp_solver": "indirect",
        "direct_n_init_nodes": 50,
        "indirect_tol": 1e-3,
        "indirect_max_nodes": 1500,
        "fp_tol": 1e-2,
        "conv_tol": 1e-2,
    }


def attach_config(config, overrides=None):
    """Attach Burgers OCP and system-specific attributes to config."""
    attach_ocp(config, BurgersOCP, get_default_config(), overrides)


class BurgersOCP(BaseOCP):
    """Burgers equation optimal control problem."""

    def __init__(self, config):
        self._config = config
        n_states = config.n_states
        n_controls = config.n_controls
        # Burgers-specific parameters (nu can be overridden for scalability / mismatch evaluation)
        self.nu = config.nu
        self.gamma = 0.1
        self.R = 0.5
        width = getattr(config, "control_width", 0.2)

        # Chebyshev + zero Dirichlet BC (interior only)
        self.xi, self.D, self.D2, self.D3, self.w_flat, self.w = _build_dirichlet_operators(n_states)

        B = spatial_control_matrix(
            self.xi, n_controls, width, domain_type="symmetric", x_min=-1.0, x_max=1.0
        )

        # Forcing term coefficient - Burgers-specific
        self.alpha = np.abs(self.xi) <= 1/5
        self.alpha = - 25.0 * self.alpha * (self.xi + 1/5)*(self.xi - 1/5)
        self.alpha = np.abs(self.alpha)
        self.alpha_flat = self.alpha.flatten()

        self.RBT = - B.T / (2.*self.R)


        # Make LQR controller
        # Linearization point
        X_bar = np.zeros((n_states, 1))
        U_bar = np.zeros((n_controls, 1))

        # Dynamics linearized around origin (dxdt ~= Ax + Bu)
        A = self.nu*self.D2 + np.diag(self.alpha_flat)

        # Cost matrices
        Q = np.diag(self.w_flat)
        R = np.diag([self.R]*n_controls)

        self.X_bar = np.reshape(X_bar, (-1,1))
        self.U_bar = np.reshape(U_bar, (-1,1))

        self.n_states = self.X_bar.shape[0]
        self.n_controls = self.U_bar.shape[0]

        self._A = np.reshape(A, (self.n_states, self.n_states))
        self._B = np.reshape(B, (self.n_states, self.n_controls))
        self._Q = np.reshape(Q, (self.n_states, self.n_states))
        self._R = np.reshape(R, (self.n_controls, self.n_controls))

        self.LQR = LQR(
            X_bar, U_bar, self._A, self._B, self._Q, self._R,
            P=None
        )

        self.B = self._B

    def forcing(self, X):
        """Shared exponential forcing: alpha * X * exp(-gamma * X)."""
        return X * self.alpha * np.exp(-self.gamma * X)

    def forcing_jacobian_diag(self, X):
        """Diagonal of d/dX [alpha * X * exp(-gamma * X)] = alpha * exp(-gamma*X) * (1 - gamma*X)."""
        return self.alpha * np.exp(-self.gamma * X) * (1.0 - self.gamma * X)

    def dynamics(self, X, U):
        """
        Burgers equation dynamics: dXdt = -0.5*D*X² + nu*D2*X + X*alpha*exp(-gamma*X) + B*U
        """
        flat_out = X.ndim < 2
        X = X.reshape(self.n_states, -1)
        U = U.reshape(self.n_controls, -1)

        dXdt = (
            -0.5 * (self.D @ X**2)
            + (self.nu * self.D2) @ X
            + self.forcing(X)
            + self.B @ U
        )
        return dXdt.flatten() if flat_out else dXdt

    def jacobians(self, X, U, F0=None):
        '''
        Burgers-specific Jacobian computation.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current states.
        U : (n_controls,) or (n_controls, n_points)  array
            Control inputs.
        F0 : ignored
            For API consistency only.

        Returns
        -------
        dFdX : (n_states, n_states) or (n_states, n_states, n_points) array
            Jacobian with respect to states, dF/dX.
        dFdU : (n_states, n_controls) or (n_states, n_controls, n_points) array
            Jacobian with respect to controls, dF/dU.
        '''
        X = X.reshape(self.n_states, -1)

        forcing_jac = self.forcing_jacobian_diag(X)

        dFdX = (
            - X * np.expand_dims(self.D, -1)
            + np.expand_dims(self.nu * self.D2, -1)
        )

        diag_idx = np.diag_indices(self.n_states)
        for k in range(X.shape[1]):
            dFdX[diag_idx[0],diag_idx[1],k] += forcing_jac[:,k]

        dFdU = np.expand_dims(self.B, -1)
        dFdU = np.tile(dFdU, (1,1,X.shape[-1]))

        return dFdX, dFdU

    def bvp_dynamics(self, t, X_aug):
        '''
        Burgers-specific augmented dynamics for Pontryagin's Minimum Principle.

        Parameters
        ----------
        X_aug : (2*n_states+1, n_points) array
            Current state, costate, and running cost.

        Returns
        -------
        dX_aug_dt : (2*n_states+1, n_points) array
            Concatenation of dynamics dXdt = F(X,U^*), costate dynamics,
            dAdt = -dH/dX(X,U^*,dVdX), and change in cost dVdt = -L(X,U*),
            where U^* is the optimal control.
        '''
        X = X_aug[:self.n_states].reshape(self.n_states, -1)
        A = X_aug[self.n_states:2*self.n_states].reshape(self.n_states, -1)

        # Control as a function of the costate
        U = self.U_star(X, A)

        wX = self.w * X

        dXdt = (
            - 0.5*np.matmul(self.D, X**2)
            + np.matmul(self.nu*self.D2, X)
            + self.forcing(X)
            + np.matmul(self.B, U)
        )

        dAdt = (
            - 2.*wX
            + X * np.matmul(self.D.T, A)
            - np.matmul(self.nu * self.D2.T, A)
            - self.forcing_jacobian_diag(X) * A
        )

        L = np.atleast_2d(self.running_cost(X, U, wX))

        return np.vstack((dXdt, dAdt, -L))

    def sample_initial_conditions(self, n: int, seed=None, **kwargs):
        """Burgers: Fourier-sine sum (unified)."""
        cfg = getattr(self, "_config", None)
        modes = kwargs.get("K") or kwargs.get("modes") or (getattr(cfg, "ic_modes", 5) if cfg else 5)
        scale = kwargs.get("scale") or (getattr(cfg, "ic_scale", 1.0) if cfg else 1.0)
        use_sine = kwargs.get("use_sine") if "use_sine" in kwargs else (getattr(cfg, "ic_use_sine", True) if cfg else True)
        use_cosine = kwargs.get("use_cosine") if "use_cosine" in kwargs else (getattr(cfg, "ic_use_cosine", False) if cfg else False)
        return sample_initial_conditions_fourier(
            self.xi, self.n_states, n, modes=modes, scale=scale,
            use_sine=use_sine, use_cosine=use_cosine, domain_type="symmetric",
            seed=seed, **{k: v for k, v in kwargs.items() if k not in ("K", "modes", "scale", "use_sine", "use_cosine")}
        )
