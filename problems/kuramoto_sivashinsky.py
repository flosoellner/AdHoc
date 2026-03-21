import numpy as np

from .base import (
    BaseOCP,
    _build_fourier_operators,
    attach_ocp,
    spatial_control_matrix,
    sample_initial_conditions_fourier,
)
from controls.lqr import LQR


def get_default_config():
    return {
        "n_controls": 4,
        "L": 2 * np.pi,
        "t1_initial": 3,
        "t1_scale": 6 / 5,
        "t1_max": 10.0,
        "n_states": 96,
        "nu": 0.25,    # strong nonlinearity: chaotic transients
        "R": 0.5,         # higher penalty: LQR is slower, chaos lives longer
        "control_width": 0.2,
        "ic_modes": 4,
        "ic_scale": 0.75,
        "ic_use_sine": True,
        "ic_use_cosine": True,
        "fp_tol": 0.05,
        "conv_tol": 0.05,
        "ocp_solver": "indirect",
        "direct_n_init_nodes": 50,
        "indirect_tol": 1e-2,
        "indirect_max_nodes": 1000,
        "seed": 42,
    }


def attach_config(config, overrides=None):
    """Attach K-S OCP and system-specific attributes to config."""
    attach_ocp(config, KSOCP, get_default_config(), overrides)



def forcing(self, X):
    return np.zeros_like(X)

def forcing_jacobian_diag(self, X):
    return np.zeros_like(X)

class KSOCP(BaseOCP):
    """
    Kuramoto-Sivashinsky equation on periodic domain [0, 2*pi).

    Following Al Jamal & Smaoui (2023), the GKS equation with alpha=1 is:
        z_t + nu * z_xxxx +  z_xx +  z * z_x = B*u 

    Linearization: A = -nu D4 - D2 - ze * D  (ze=0 here)
                     = -nu * D4 - D2 + diag(alpha)

    Number of unstable eigenvalues: N_unstable = floor(sqrt(1/nu)) 
    (n=0 is neutral, n=+/-1 are unstable).
    """

    def __init__(self, config):
        self._config = config
        n_states = config.n_states
        n_controls = config.n_controls
        self.nu = config.nu
        self.L = config.L

        # Fourier spectral operators on [0, L) periodic
        self.xi, self.D, self.D2, self.D3, self.w_flat, self.w = \
            _build_fourier_operators(n_states, self.L)

        # Fourth-order derivative
        self.D4 = self.D2 @ self.D2

        xi_flat = self.xi.flatten()

        width = getattr(config, "control_width", 0.25)
        B = spatial_control_matrix(
            self.xi, n_controls, width, domain_type="periodic", L=self.L
        )



        # Scalar control penalty
        self.R = config.R
        self.RBT = -B.T / (2.0 * self.R)

        # Linearization around zero:
        # A = -nu*D4 - D2 + diag(alpha)
        # (the z*D term vanishes at ze=0)
        A_lin = -self.nu * self.D4 - self.D2

        # Cost matrices
        Q = np.diag(self.w_flat.flatten() / self.L) 
        R_mat = np.diag([self.R] * n_controls)

        X_bar = np.zeros((n_states, 1))
        U_bar = np.zeros((n_controls, 1))

        self.X_bar = np.reshape(X_bar, (-1, 1))
        self.U_bar = np.reshape(U_bar, (-1, 1))
        self.n_states = self.X_bar.shape[0]
        self.n_controls = self.U_bar.shape[0]

        self._A = np.reshape(A_lin, (self.n_states, self.n_states))
        self._B = np.reshape(B,     (self.n_states, self.n_controls))
        self._Q = np.reshape(Q,     (self.n_states, self.n_states))
        self._R = np.reshape(R_mat, (self.n_controls, self.n_controls))

        self.LQR = LQR(self.X_bar, self.U_bar, self._A, self._B, self._Q, self._R)

        self.B = self._B

        # --- Diagnostic: instability spectrum ---
        A_no_alpha = -self.nu * self.D4 - self.D2
        eigs_ks = np.sort(np.linalg.eigvals(A_no_alpha).real)
        eigs_full = np.sort(np.linalg.eigvals(self._A).real)
        print(f"KS only (top 5):  {eigs_ks[-5:]}")
        print(f"KS + alpha (top 5): {eigs_full[-5:]}")
        print(f"Number unstable (KS only):  {np.sum(eigs_ks > 0)}")
        print(f"Number unstable (KS + alpha): {np.sum(eigs_full > 0)}")

        K_lqr = self.LQR.K
        A_cl = self._A - self._B @ K_lqr
        eigs_cl = np.sort(np.linalg.eigvals(A_cl).real)
        print(f"Closed-loop (top 5): {eigs_cl[-5:]}")
        print(f"Closed-loop stable: {np.all(eigs_cl < 0)}")

        # Stabilizability check
        eigs, vecs = np.linalg.eig(self._A)
        unstable_idx = np.where(eigs.real > 1e-10)[0]
        V_unstable = vecs[:, unstable_idx].real
        C_tilde = V_unstable.T @ self._B
        print(f"Stabilizability matrix rank: {np.linalg.matrix_rank(C_tilde)} / {len(unstable_idx)}")

    def dynamics(self, X, U):
        """
        GKS dynamics (alpha=1):
          dXdt = -nu*D4*X - D2*X - 0.5*D(X^2) 
        """
        flat_out = X.ndim < 2
        X = X.reshape(self.n_states, -1)
        U = U.reshape(self.n_controls, -1)

        dXdt = (
            -self.nu * (self.D4 @ X)
            -  (self.D2 @ X)
            -  0.5 * (self.D @ (X ** 2))
            + self.B @ U
        )
        return dXdt.flatten() if flat_out else dXdt

    def jacobians(self, X, U, F0=None):
        """
        GKS Jacobians (alpha=1).
        df/dX = -nu*D4 - D2 - (diag(DX) + diag(X)*D)

        df/dU = B
        """
        X = X.reshape(self.n_states, -1)
        N = X.shape[1]
        DX = self.D @ X

        linear = (-self.nu * self.D4 - self.D2)[:, :, None]
        dFdX = np.broadcast_to(linear, (self.n_states, self.n_states, N)).copy()

        diag_idx = np.diag_indices(self.n_states)
        dFdX[diag_idx[0], diag_idx[1], :] -= DX
        dFdX -= X[:, None, :] * self.D[:, :, None]



        dFdU = np.tile(self.B[:, :, None], (1, 1, N))

        return dFdX, dFdU

    def bvp_dynamics(self, t, X_aug):
        """
        Augmented PMP dynamics for solve_bvp (state + costate + cost).
        """
        X = X_aug[:self.n_states].reshape(self.n_states, -1)
        A = X_aug[self.n_states:2 * self.n_states].reshape(self.n_states, -1)

        U = self.U_star(X, A)

        wX = self.w * X

        dXdt = (
            -self.nu * (self.D4 @ X)
            - (self.D2 @ X)
            -  0.5 * (self.D @ (X ** 2))
            + self.B @ U
        )

        DX = self.D @ X

        dAdt = (
            - 2.0 * wX
            + self.nu * (self.D4.T @ A)
            + (self.D2.T @ A)
            + DX * A
            + X * (self.D.T @ A)
        )

        L = np.atleast_2d(self.running_cost(X, U, wX))

        return np.vstack((dXdt, dAdt, -L))

    def sample_initial_conditions(self, n: int, seed=None, **kwargs):
        """KS: Fourier sine+cosine sum (unified)."""
        cfg = self._config
        modes = kwargs.get("K") or kwargs.get("modes") or getattr(cfg, "ic_modes", 3)
        scale = kwargs.get("scale") or getattr(cfg, "ic_scale", 0.1)
        use_sine = kwargs.get("use_sine") if "use_sine" in kwargs else getattr(cfg, "ic_use_sine", True)
        use_cosine = kwargs.get("use_cosine") if "use_cosine" in kwargs else getattr(cfg, "ic_use_cosine", True)
        return sample_initial_conditions_fourier(
            self.xi, self.n_states, n, modes=modes, scale=scale,
            use_sine=use_sine, use_cosine=use_cosine, domain_type="periodic", L=self.L,
            seed=seed, **{k: v for k, v in kwargs.items() if k not in ("K", "modes", "scale", "use_sine", "use_cosine")}
        )