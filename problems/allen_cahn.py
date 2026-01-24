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
    
    # Effective D1 on interior: D1_II y_I + D1_IB y_B (for advection)
    D1_IB = D1_full[np.ix_(Int, Bnd)]    # (n,2)
    D1_II = D1_full[np.ix_(Int, Int)]    # (n,n)
    D1_eff = D1_II + D1_IB @ E           # (n,n)

    xi = x_nodes[Int].reshape(-1, 1)     # (n,1)
    w_flat = w_full[Int]                 # (n,)
    w = w_flat.reshape(-1, 1)            # (n,1)
    D1_int = D1_full[np.ix_(Int, Int)]   # (n,n) (not strictly needed for Allen–Cahn)

    return xi, D1_int, D1_eff, D2_eff, w_flat, w


class AllenCahnOCP(BaseOCP):
    """Allen–Cahn optimal control problem (Neumann BC discretization)."""

    def __init__(self, config):
        n_states = int(config.n_states)
        n_controls = int(config.n_controls)

        # physics params (match your legacy defaults)
        self.nu = 0.1
        self.R = 0.5

        # Perturbation parameters (can be set via config)
        # perturbation_type: None, 'exponential', or 'advection'
        self.perturbation_type = getattr(config, 'perturbation_type', None)
        self.perturbation_strength = getattr(config, 'perturbation_strength', 1.0)
        # For exponential: perturbation_strength is the coefficient
        # For advection: perturbation_strength is the advection velocity c

        # operators + quadrature weights
        self.xi, self.D, self.D1, self.D2, self.w_flat, self.w = _build_neumann_operators(n_states)

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

        # linearization point - target stable steady state (-1 or 1)
        # Using -1 as default (both -1 and 1 are stable)
        target_value = 0 # -1.0 or 1.0
        X_bar = np.full((n_states, 1), target_value)
        U_bar = np.zeros((n_controls, 1))

        # linearized dynamics at x = X_bar (stable steady state)
        # d/dx (x - x^3) = 1 - 3x^2
        # At x = -1: 1 - 3(-1)^2 = 1 - 3 = -2
        # At x = 1:  1 - 3(1)^2 = 1 - 3 = -2
        # So A = nu*D2 + diag(-2) = nu*D2 - 2*I (stable!)
        diag_term = 1.0 - 3.0 * (target_value ** 2)  # = -2 for ±1
        A = self.nu * self.D2 + diag_term * np.eye(n_states)
        
        # Add perturbation contribution to linearization
        if self.perturbation_type == 'exponential':
            # d/dx [-c * x * exp(-0.5 * x)] = -c * exp(-0.5*x) * (1 - 0.5*x)
            # At x = target_value:
            exp_term = np.exp(-0.5 * target_value)
            diag_pert = -self.perturbation_strength * exp_term * (1.0 - 0.5 * target_value)
            A += diag_pert * np.eye(n_states)
        elif self.perturbation_type == 'advection':
            # Advection: -c * D1
            A -= self.perturbation_strength * self.D1 

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

    def running_cost(self, X, U, wX=None):
        """
        Running cost L(X,U) = ||X - X_bar||^2_w + R||U||^2
        Penalizes distance from target steady state X_bar.
        """
        X = X.reshape(self.n_states, -1)
        U = U.reshape(self.n_controls, -1)
        
        # Compute X - X_bar
        X_err = X - self.X_bar
        
        if wX is None:
            if X.ndim == 1:
                wX_err = self.w_flat * X_err.flatten()
            else:
                wX_err = self.w * X_err
        else:
            # If wX provided, it should be w * X, so convert to w * (X - X_bar)
            wX_err = wX - (self.w * self.X_bar)
        
        return np.sum(wX_err * X_err, axis=0) + self.R * np.sum(U**2, axis=0)

    def running_cost_gradient(self, X, U, return_dLdX=True, return_dLdU=True):
        """
        Gradient of running cost: dL/dX = 2*w*(X - X_bar), dL/dU = 2*R*U
        """
        X = X.reshape(self.n_states, -1)
        U = U.reshape(self.n_controls, -1)
        X_err = X - self.X_bar
        
        dLdX = None
        dLdU = None
        
        if return_dLdX:
            if X.ndim == 1:
                dLdX = 2.0 * (self.w_flat * X_err.flatten())
            else:
                dLdX = 2.0 * (self.w * X_err)
            if not return_dLdU:
                return dLdX
        
        if return_dLdU:
            dLdU = 2.0 * self.R * U
            if not return_dLdX:
                return dLdU
        
        return dLdX, dLdU

    def dynamics(self, X, U):
        """
        Allen–Cahn:
          dXdt = nu * D2 * X + X - X^3 + B * U + perturbation
        
        Perturbations:
        - 'exponential': -perturbation_strength * X * exp(-0.5 * X)
        - 'advection': -perturbation_strength * D1 * X
        """
        flat_out = X.ndim < 2
        X = X.reshape(self.n_states, -1)
        U = U.reshape(self.n_controls, -1)

        dXdt = (self.nu * (self.D2 @ X)) + X - (X ** 3) + (self.B @ U)
        
        # Add perturbation if specified
        if self.perturbation_type == 'exponential':
            # Exponential perturbation: -c * X * exp(-0.5 * X)
            perturbation = -self.perturbation_strength * X * np.exp(-0.5 * X)
            dXdt += perturbation
        elif self.perturbation_type == 'advection':
            # Advection perturbation: -c * D1 * X
            perturbation = -self.perturbation_strength * (self.D1 @ X)
            dXdt += perturbation
        
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
            
            # Add perturbation Jacobian
            if self.perturbation_type == 'exponential':
                # d/dx [-c * x * exp(-0.5 * x)] = -c * [exp(-0.5*x) - 0.5*x*exp(-0.5*x)]
                # = -c * exp(-0.5*x) * (1 - 0.5*x)
                exp_term = np.exp(-0.5 * X[:, k])
                diag_pert = -self.perturbation_strength * exp_term * (1.0 - 0.5 * X[:, k])
                J[idx, idx] += diag_pert
            elif self.perturbation_type == 'advection':
                # Advection: -c * D1, so Jacobian is -c * D1
                J -= self.perturbation_strength * self.D1
            
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

        # dynamics: dXdt = nu D2 X + X - X^3 + B U + perturbation
        dXdt = (self.nu * (self.D2 @ X)) + X - (X ** 3) + (self.B @ U)
        
        # Add perturbation if specified
        if self.perturbation_type == 'exponential':
            perturbation = -self.perturbation_strength * X * np.exp(-0.5 * X)
            dXdt += perturbation
        elif self.perturbation_type == 'advection':
            perturbation = -self.perturbation_strength * (self.D1 @ X)
            dXdt += perturbation

        # costate dynamics: dA/dt = -∂H/∂X = -∂L/∂X - (∂f/∂X)^T A
        # Here L penalizes distance from X_bar => dL/dX = 2 w (X - X_bar)
        X_err = X - self.X_bar                 # (n,N)
        wX_err = self.w * X_err                # (n,N)
        dLdX = 2.0 * wX_err                    # (n,N)

        # ∂f/∂X = nu D2 + diag(1 - 3 X^2) + perturbation_jacobian
        # so (∂f/∂X)^T A = (nu D2^T) A + diag(1 - 3X^2) A + perturbation_jacobian^T A
        dAdt = -dLdX - (self.nu * (self.D2.T @ A)) - ((1.0 - 3.0 * (X ** 2)) * A)
        
        # Add perturbation contribution to costate dynamics
        if self.perturbation_type == 'exponential':
            # d/dx [-c * x * exp(-0.5 * x)] = -c * exp(-0.5*x) * (1 - 0.5*x)
            exp_term = np.exp(-0.5 * X)
            diag_pert = -self.perturbation_strength * exp_term * (1.0 - 0.5 * X)
            dAdt -= diag_pert * A
        elif self.perturbation_type == 'advection':
            # Advection: -c * D1, so (∂f/∂X)^T = -c * D1^T
            dAdt -= self.perturbation_strength * (self.D1.T @ A)

        # running cost (1,N) - pass wX_err for consistency
        L = np.atleast_2d(self.running_cost(X, U, wX_err))

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
        self.register_buffer("X_bar", torch.from_numpy(ocp.X_bar).float())

        self.nu = float(ocp.nu)
        self.R_val = float(ocp.R)
        
        # Store perturbation parameters
        self.perturbation_type = ocp.perturbation_type
        self.perturbation_strength = float(ocp.perturbation_strength)
        
        # Store D1 for advection if needed
        if ocp.perturbation_type == 'advection':
            self.register_buffer("D1", torch.from_numpy(ocp.D1).float())
        else:
            self.D1 = None

    def get_control(self, dVdX: torch.Tensor) -> torch.Tensor:
        # u = -(1/(2R)) B^T dVdX  ==  -(0.5/R) (dVdX @ B)
        u = -(0.5 / self.R_val) * (dVdX @ self.B)
        return torch.clamp(u, -10, 10)

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_T = x.t()  # (n,B)
        f = self.nu * (self.D2 @ x_T) + x_T - (x_T ** 3) + (self.B @ u.t())
        
        # Add perturbation if specified
        if self.perturbation_type == 'exponential':
            # Exponential perturbation: -c * X * exp(-0.5 * X)
            perturbation = -self.perturbation_strength * x_T * torch.exp(-0.5 * x_T)
            f += perturbation
        elif self.perturbation_type == 'advection':
            # Advection perturbation: -c * D1 * X
            perturbation = -self.perturbation_strength * (self.D1 @ x_T)
            f += perturbation
        
        return f.t()

    def running_cost(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Penalize distance from X_bar (stable steady state)
        # x is (B, n), X_bar is (n_states, 1), need to reshape to (1, n) for broadcasting
        X_bar_flat = self.X_bar.squeeze(1)  # (n_states,)
        x_err = x - X_bar_flat.unsqueeze(0)  # (B, n)
        state_cost = (self.w * (x_err ** 2)).sum(dim=1)
        control_cost = self.R_val * (u ** 2).sum(dim=1)
        return state_cost + control_cost