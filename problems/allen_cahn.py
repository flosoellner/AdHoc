import numpy as np
import torch

from .base import BaseOCP, cheb
from controls.lqr import LQR


def _build_neumann_operators(n_states: int):
    """
    Build Chebyshev collocation operators on [-1,1] with homogeneous Neumann BC,
    returning interior operators of size (n_states, n_states).

    We build a full grid of size n_full = n_states + 2, then eliminate boundary
    values via Neumann constraints D1 y = 0 at both ends.
    """
    n_full = n_states + 2
    x_nodes, D1_full, w_full = cheb(n_full - 1)  # -> (n_full,)
    D2_full = D1_full @ D1_full

    Bnd = np.array([0, n_full - 1])      # boundary indices
    Int = np.arange(1, n_full - 1)       # interior indices (size n_states)

    D1_BB = D1_full[np.ix_(Bnd, Bnd)]    # (2,2)
    D1_BI = D1_full[np.ix_(Bnd, Int)]    # (2,n)
    D2_IB = D2_full[np.ix_(Int, Bnd)]    # (n,2)
    D2_II = D2_full[np.ix_(Int, Int)]    # (n,n)

    # Neumann: D1_BB y_B + D1_BI y_I = 0  ->  y_B = E y_I
    E = -np.linalg.solve(D1_BB, D1_BI)   # (2,n)

    # Effective D2 on interior: D2_II y_I + D2_IB y_B
    D2_eff = D2_II + D2_IB @ E           # (n,n)

    xi = x_nodes[Int].reshape(-1, 1)     # (n,1)
    w_flat = w_full[Int]                 # (n,)
    w = w_flat.reshape(-1, 1)            # (n,1)
    D1_int = D1_full[np.ix_(Int, Int)]   # (n,n) (not strictly needed for Allen–Cahn)

    return xi, D1_int, D2_eff, w_flat, w


class AllenCahnOCP(BaseOCP):
    """Allen–Cahn optimal control problem (Neumann BC discretization)."""

    def __init__(self, config):
        n_states = int(config.n_states)
        n_controls = int(config.n_controls)

        # physics params (match your legacy defaults)
        self.nu = 0.017
        self.R = 0.5

        # operators + quadrature weights
        self.xi, self.D, self.D2, self.w_flat, self.w = _build_neumann_operators(n_states)

        # control matrix B (simple default; adjust as you like)
        # If m=3, use the legacy “3 regions”; else split indices evenly.
        if n_controls == 3:
            B = np.zeros((n_states, 3), dtype=float)
            xi_flat = self.xi.flatten()
            w1 = (xi_flat > -0.7) & (xi_flat < -0.4)
            w2 = (xi_flat > -0.2) & (xi_flat < 0.2)
            w3 = (xi_flat > 0.4) & (xi_flat < 0.7)
            B[w1, 0] = 1.0
            B[w2, 1] = 1.0
            B[w3, 2] = 1.0
        else:
            B = np.zeros((n_states, n_controls), dtype=float)
            parts = np.array_split(np.arange(n_states), n_controls)
            for j, idx in enumerate(parts):
                B[idx, j] = 1.0

        # linearization point
        X_bar = np.zeros((n_states, 1))
        U_bar = np.zeros((n_controls, 1))

        # linearized dynamics at x=0:  x_t = nu D2 x + x + B u
        A = self.nu * self.D2 + np.eye(n_states)

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
          dXdt = nu * D2 * X + X - X^3 + B * U
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

        # dynamics: dXdt = nu D2 X + X - X^3 + B U
        dXdt = (self.nu * (self.D2 @ X)) + X - (X ** 3) + (self.B @ U)

        # costate dynamics: dA/dt = -∂H/∂X = -∂L/∂X - (∂f/∂X)^T A
        # Here L uses BaseOCP.running_cost => dL/dX = 2 w X (w is quadrature weights)
        wX = self.w * X                        # (n,N)
        dLdX = 2.0 * wX                        # (n,N)

        # ∂f/∂X = nu D2 + diag(1 - 3 X^2)
        # so (∂f/∂X)^T A = (nu D2^T) A + diag(1 - 3X^2) A
        dAdt = -dLdX - (self.nu * (self.D2.T @ A)) - ((1.0 - 3.0 * (X ** 2)) * A)

        # running cost (1,N)
        L = np.atleast_2d(self.running_cost(X, U, wX))

        return np.vstack((dXdt, dAdt, -L))

    def physics_module(self):
        return AllenCahnPhysics(self)


class AllenCahnPhysics(torch.nn.Module):
    """Differentiable Allen–Cahn dynamics + running cost (torch-only)."""

    def __init__(self, ocp: AllenCahnOCP):
        super().__init__()
        self.register_buffer("D2", torch.from_numpy(ocp.D2).float())
        self.register_buffer("B", torch.from_numpy(ocp.B).float())
        self.register_buffer("w", torch.from_numpy(ocp.w_flat).float())

        self.nu = float(ocp.nu)
        self.R_val = float(ocp.R)

    def get_control(self, dVdX: torch.Tensor) -> torch.Tensor:
        # u = -(1/(2R)) B^T dVdX  ==  -(0.5/R) (dVdX @ B)
        u = -(0.5 / self.R_val) * (dVdX @ self.B)
        return torch.clamp(u, -0.5, 0.5)

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_T = x.t()  # (n,B)
        f = self.nu * (self.D2 @ x_T) + x_T - (x_T ** 3) + (self.B @ u.t())
        return f.t()

    def running_cost(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        state_cost = (self.w * (x ** 2)).sum(dim=1)
        control_cost = self.R_val * (u ** 2).sum(dim=1)
        return state_cost + control_cost