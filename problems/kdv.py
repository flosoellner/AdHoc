import numpy as np

from .base import BaseOCP, _build_full_grid_operators
from controls.lqr import LQR


def get_default_config():
    """Kuramoto–Sivashinsky system default config (in kdv.py).
    PDE: u_t + nu4*u_xxxx + u_xx + u·u_x = control on (0,1). nu4 in (0,1] scales D4 (stiffness).
    BCs: u_xx(0)=k·u_x(0), u_xxx(0)=-k·u(0)-u(0)³; u_xx(1)=-k·u_x(1), u_xxx(1)=k·u(1)+u(1)³.
    """
    return {
        "n_states": 32,
        "n_controls": 2,
        "k_bc": 1.0,
        "nu4": 10.0,
        "t1_initial": 15.0,
        "t1_scale": 1.2,
        "t1_max": 60.0,
        "ocp_solver": "indirect",
        "direct_n_init_nodes": 50,
        "indirect_tol": 5e-3,
        "indirect_max_nodes": 2000,
    }

def attach_config(config, overrides=None):
    """Attach Kuramoto–Sivashinsky OCP and system-specific attributes to config."""
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
    """Kuramoto–Sivashinsky OCP on (0,1): u_t + u_xxxx + u_xx + u·u_x = control.
    BCs: u_xx(0)=k·u_x(0), u_xxx(0)=-k·u(0)-u(0)³; u_xx(1)=-k·u_x(1), u_xxx(1)=k·u(1)+u(1)³."""
    def __init__(self, config):
        n_states = config.n_states
        n_controls = config.n_controls
        k = getattr(config, "k_bc", 1.0)
        nu4 = getattr(config, "nu4", 1.0)

        self.R = 1.0
        self.k_bc = k
        self.nu4 = nu4

        self.xi, self.D, self.D2, self.D3, self.D4, self.w_flat, self.w = _build_full_grid_operators(n_states)
        # D2 with BC rows: u_xx(0)=k*u_x(0), u_xx(1)=-k*u_x(1)
        self.D2_bc = self.D2.copy()
        self.D2_bc[0, :] = k * self.D[0, :]
        self.D2_bc[-1, :] = -k * self.D[-1, :]

        # Actuators (Simplified)
        self.B = np.zeros((n_states, n_controls))
        self.B[:n_states//2, 0] = 1.0
        self.B[n_states//2:, 1] = 1.0
        self.RBT = - self.B.T / (2. * self.R)

        # LQR: linearization at zero; A_lin = -D2_bc - nu4*D4 (nu4 scales stiffness)
        X_bar = np.zeros((n_states, 1))
        U_bar = np.zeros((n_controls, 1))
        A_lin = -self.D2_bc - nu4 * self.D4
        Q_lqr = np.diag(self.w_flat)
        R_lqr = np.diag([self.R] * n_controls)

        self.X_bar, self.U_bar = X_bar, U_bar
        self.n_states, self.n_controls = n_states, n_controls
        try:
            self.LQR = LQR(X_bar, U_bar, A_lin, self.B, Q_lqr, R_lqr)
        except np.linalg.LinAlgError:
            # CARE can fail when Hamiltonian pencil has eigenvalues near imaginary axis
            P_fallback = Q_lqr + 1e-2 * np.eye(n_states)
            self.LQR = LQR(X_bar, U_bar, A_lin, self.B, Q_lqr, R_lqr, P=P_fallback)

    def dynamics(self, X, U):
        """
        KS: u_t + nu4*u_xxxx + u_xx + u·u_x = control on (0,1).
        Interior: dX/dt = -nu4*(D4@X) - (D2_bc@X) - X*(D@X) + B@U.
        Boundary rows overwritten using u_xxxx from BC (formulas depend on nu4).
        """
        flat_out = X.ndim < 2
        X = X.reshape(self.n_states, -1)
        U = U.reshape(self.n_controls, -1)
        k, nu4 = self.k_bc, self.nu4

        Dx = self.D @ X
        dXdt = (
            - nu4 * (self.D4 @ X)
            - (self.D2_bc @ X)
            - X * Dx
            + self.B @ U
        )
        # u_t(0) = (k*(nu4-1) + 3*nu4*u_0² - u_0)*u_x(0) + ctrl, u_t(1) = (k*(1-nu4) - 3*nu4*u_1² - u_1)*u_x(1) + ctrl
        n_cols = X.shape[1]
        for j in range(n_cols):
            u0, u1 = X[0, j], X[-1, j]
            dx0, dx1 = Dx[0, j], Dx[-1, j]
            dXdt[0, j] = (k * (nu4 - 1) + 3 * nu4 * u0**2 - u0) * dx0 + (self.B @ U[:, j:j+1])[0, 0]
            dXdt[-1, j] = (k * (1 - nu4) - 3 * nu4 * u1**2 - u1) * dx1 + (self.B @ U[:, j:j+1])[-1, 0]
        return dXdt.flatten() if flat_out else dXdt

    def jacobians(self, X, U, F0=None):
        X = X.reshape(self.n_states, -1)
        Dx = self.D @ X
        k, nu4 = self.k_bc, self.nu4
        # Interior-like: -D2_bc - nu4*D4 - (diag(Dx) + diag(X)@D)
        base_jac = -self.D2_bc - nu4 * self.D4

        dFdX = np.zeros((self.n_states, self.n_states, X.shape[1]))
        for col in range(X.shape[1]):
            dFdX[:, :, col] = base_jac - (np.diag(Dx[:, col]) + np.diag(X[:, col]) @ self.D)
            # Boundary rows: d/dX of (k*(nu4-1)+3*nu4*u_0^2-u_0)*Dx_0 and (k*(1-nu4)-3*nu4*u_1^2-u_1)*Dx_1
            u0, u1 = X[0, col], X[-1, col]
            dx0, dx1 = Dx[0, col], Dx[-1, col]
            dFdX[0, :, col] = (6 * nu4 * u0 - 1) * dx0 * (np.arange(self.n_states) == 0).astype(float) + (k * (nu4 - 1) + 3 * nu4 * u0**2 - u0) * self.D[0, :]
            dFdX[-1, :, col] = (-6 * nu4 * u1 - 1) * dx1 * (np.arange(self.n_states) == self.n_states - 1).astype(float) + (k * (1 - nu4) - 3 * nu4 * u1**2 - u1) * self.D[-1, :]

        dFdU = np.tile(np.expand_dims(self.B, -1), (1, 1, X.shape[-1]))
        return dFdX, dFdU

    def bvp_dynamics(self, t, X_aug):
        X = X_aug[:self.n_states].reshape(self.n_states, -1)
        A = X_aug[self.n_states:2*self.n_states].reshape(self.n_states, -1)

        U = self.U_star(X, A)
        wX = self.w * X

        # State: KS with nu4 and boundary overwrite (same as dynamics)
        k, nu4 = self.k_bc, self.nu4
        Dx = self.D @ X
        dXdt = - nu4 * (self.D4 @ X) - (self.D2_bc @ X) - X * Dx + self.B @ U
        n_cols = X.shape[1]
        for j in range(n_cols):
            u0, u1 = X[0, j], X[-1, j]
            dx0, dx1 = Dx[0, j], Dx[-1, j]
            dXdt[0, j] = (k * (nu4 - 1) + 3 * nu4 * u0**2 - u0) * dx0 + (self.B @ U[:, j:j+1])[0, 0]
            dXdt[-1, j] = (k * (1 - nu4) - 3 * nu4 * u1**2 - u1) * dx1 + (self.B @ U[:, j:j+1])[-1, 0]

        # Costate: dAdt = -dL/dX - (df/dX)' A; use actual Jacobian (with BC rows)
        dFdX, _ = self.jacobians(X, U)
        dAdt = np.zeros_like(A)
        for j in range(A.shape[1]):
            dAdt[:, j] = -2.0 * wX[:, j] - dFdX[:, :, j].T @ A[:, j]

        L = np.atleast_2d(self.running_cost(X, U, wX))
        return np.vstack((dXdt, dAdt, -L))
        


    def sample_initial_conditions(self, n: int, seed=None, **kwargs):
        if seed is not None: np.random.seed(seed)
        xi = self.xi.flatten()  # [0, 1]
        X0 = np.zeros((self.n_states, n))
        for i in range(n):
            amp = 0.1
            width = 0.2
            center = 0.5
            X0[:, i] = amp * np.exp(-((xi - center)**2) / (2 * width**2))
        return X0