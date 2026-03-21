import numpy as np
import warnings

# Suppress RuntimeWarnings for overflow/invalid value operations
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')

# Suppress NumPy warnings at the NumPy level (more reliable)
np.seterr(over='ignore', invalid='ignore')

try:
    from scipy.integrate import cumtrapz
except:
    from scipy.integrate import cumulative_trapezoid as cumtrapz

from scipy.optimize._numdiff import approx_derivative
from scipy import sparse

try:
    import torch
except ImportError:
    torch = None


def u_analytic(Vx: np.ndarray, config) -> np.ndarray:
    """
    Compute optimal control from value gradient using PMP.
    
    Parameters:
    -----------
    Vx : array, shape (d,)
        Value gradient (costate)
    config : config-like object
        Must have: q (control penalty scalar or array), B (control matrix)
        
    Returns:
    --------
    u_optimal : array, shape (m,)
        Optimal control
    """
    q = config.q
    B = config.B

    m = B.shape[1]
    if np.isscalar(q):
        R_inv = np.eye(m) / (2.0 * q)
    else:
        R_inv = np.diag(1.0 / (2.0 * np.asarray(q).flatten()))
    u_optimal = -R_inv @ (B.T @ Vx)

    return u_optimal

def find_fixed_point(OCP, controller, fp_tol, X0=None, verbose=True):
    '''
    Use root-finding to find a fixed point (equilibrium) of the closed-loop
    dynamics near the desired goal state OCP.X_bar. Also computes the
    closed-loop Jacobian and its eigenvalues.

    Parameters
    ----------
    OCP : instance of BaseOCP or subclass
    controller : controller instance
    tol : float
        Maximum value of the vector field allowed for a trajectory to be
        considered as convergence to an equilibrium
    X0 : array, optional
        Initial guess for the fixed point. If X0=None, use OCP.X_bar
    verbose : bool, default=True
        Set to True to print out the deviation of the fixed point from OCP.X_bar
        and the Jacobian eigenvalue

    Returns
    -------
    X_star : (n_states, 1) array
        Closed-loop equilibrium
    X_star_err : float
        ||X_star - OCP.X_bar||
    F_star : (n_states, 1) array
        Vector field evaluated at X_star. If successful should have F_star ~ 0
    Jac : (n_states, n_states) array
        Closed-loop Jacobian at X_star
    eigs : (n_states, 1) complex array
        Eigenvalues of the closed-loop Jacobian
    max_eig : complex scalar
        Largest eigenvalue of the closed-loop Jacobian
    '''
    if X0 is None:
        X0 = OCP.X_bar
    X0 = np.reshape(X0, (OCP.n_states,))

    def dynamics_wrapper(X):
        U = controller.eval_U(X)
        F = OCP.dynamics(X, U)
        C = OCP.constraint_fun(X)
        if C is not None:
            F = np.concatenate((F.flatten(), C.flatten()))
        return F

    def Jacobian_wrapper(X):
        J = OCP.closed_loop_jacobian(X, controller)
        JC = OCP.constraint_jacobian(X)
        if JC is not None:
            J = np.vstack((
                J.reshape(-1,X.shape[0]), JC.reshape(-1,X.shape[0])
            ))
        return J

    from scipy.optimize import root
    sol = root(dynamics_wrapper, X0, jac=Jacobian_wrapper, method='lm')

    sol.x = OCP.apply_state_constraints(sol.x)

    X_star = sol.x.reshape(-1,1)
    U_star = controller.eval_U(X_star)
    F_star = OCP.dynamics(X_star, U_star).reshape(-1,1)
    Jac = OCP.closed_loop_jacobian(sol.x, controller)

    X_star_err = OCP.norm(X_star)[0]

    eigs = np.linalg.eigvals(Jac)
    idx = np.argsort(eigs.real)
    eigs = eigs[idx].reshape(-1,1)
    max_eig = np.squeeze(eigs[-1])

    # Some linearized systems always have one or more zero eigenvalues.
    # Handle this situation by taking the next largest.
    if np.abs(max_eig.real) < fp_tol**2:
        Jac0 = np.squeeze(OCP.closed_loop_jacobian(OCP.X_bar, OCP.LQR))
        eigs0 = np.linalg.eigvals(Jac0)
        idx = np.argsort(eigs0.real)
        eigs0 = eigs0[idx].reshape(-1,1)
        max_eig0 = np.squeeze(eigs0[-1])

        i = 2
        while all([
                i <= OCP.n_states,
                np.abs(max_eig.real) < fp_tol**2,
                np.abs(max_eig0.real) < fp_tol**2
            ]):
            max_eig = np.squeeze(eigs[OCP.n_states - i])
            max_eig0 = np.squeeze(eigs0[OCP.n_states - i])
            i += 1

    if verbose:
        s = '||actual - desired_equilibrium|| = {norm:1.2e}'
        print(s.format(norm=X_star_err))
        if np.max(np.abs(F_star)) > fp_tol:
            print('Dynamics f(X_star):')
            print(F_star)
        s = 'Largest Jacobian eigenvalue = {real:1.2e} + j{imag:1.2e} \n'
        print(s.format(real=max_eig.real, imag=np.abs(max_eig.imag)))

    return X_star, X_star_err, F_star, Jac, eigs, max_eig


def cheb(N):
    '''
    Build Chebyshev differentiation matrix.
    Uses algorithm on page 54 of Spectral Methods in MATLAB by Trefethen.
    '''
    theta = np.pi / N * np.arange(0, N+1)
    X_nodes = np.cos(theta)

    X = np.tile(X_nodes, (N+1, 1))
    X = X.T - X

    C = np.concatenate(([2.], np.ones(N-1), [2.]))
    C[1::2] = -C[1::2]
    C = np.outer(C, 1./C)

    D = C / (X + np.identity(N+1))
    D = D - np.diag(D.sum(axis=1))

    # Clenshaw-Curtis weights
    # Uses algorithm on page 128 of Spectral Methods in MATLAB
    w = np.empty_like(X_nodes)
    v = np.ones(N-1)
    for k in range(2, N, 2):
        v -= 2.*np.cos(k * theta[1:-1]) / (k**2 - 1)

    if N % 2 == 0:
        w[0] = 1./(N**2 - 1)
        v -= np.cos(N*theta[1:-1]) / (N**2 - 1)
    else:
        w[0] = 1./N**2

    w[-1] = w[0]
    w[1:-1] = 2.*v/N

    return X_nodes, D, w


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
    w_flat = w_full[Int]                  # (n,)
    w = w_flat.reshape(-1, 1)            # (n,1)
    D1_int = D1_full[np.ix_(Int, Int)]   # (n,n)

    return xi, D1_int, D1_eff, D2_eff, w_flat, w



def _build_dirichlet_operators(n_states: int):
    """
    Build Chebyshev operators with zero Dirichlet BC.
    CRITICAL: Build D2 and D3 on the FULL grid before truncating to interior.
    """
    # Build full grid (n_states interior + 2 boundary nodes)
    x_nodes, D_full, w_full = cheb(n_states + 1)
    
    # Compute the physics on the full physical grid first
    D2_full = D_full @ D_full
    D3_full = D_full @ D2_full

    # Interior indices (dropping the first and last nodes)
    xi = x_nodes[1:-1].reshape(-1, 1)
    w_flat = w_full[1:-1]
    w = w_flat.reshape(-1, 1)
    
    # Now truncate to get the interior operators
    D = D_full[1:-1, 1:-1]
    D2 = D2_full[1:-1, 1:-1]
    D3 = D3_full[1:-1, 1:-1]

    return xi, D, D2, D3, w_flat, w


def _build_fourier_operators(n_states: int, L: float = 2 * np.pi):
    """
    Build Fourier spectral differentiation matrices on [0, L) with periodic BC.

    Parameters
    ----------
    n_states : int
        Number of equispaced grid points (must be even for clean FFT symmetry).
    L : float
        Domain length (default 2*pi).

    Returns
    -------
    xi : (n,1) array – grid points on [0, L).
    D  : (n,n) array – first derivative matrix.
    D2 : (n,n) array – second derivative matrix.
    D3 : (n,n) array – third derivative matrix (unused by K-S, included for interface).
    w_flat : (n,) array – quadrature weights (uniform, L/n each).
    w : (n,1) array – column version of w_flat.
    """
    N = n_states
    h = L / N
    xi = np.arange(N) * h                         # [0, h, 2h, ..., (N-1)h)
    # Wavenumbers for real FFT ordering
    k = (2 * np.pi / L) * np.fft.fftfreq(N, d=1.0 / N)  # [0, 1, 2, ..., N/2-1, -N/2+1, ..., -1] * 2pi/L

    # Build differentiation matrices via F^{-1} diag(ik)^p F
    F = np.fft.fft(np.eye(N), axis=0)             # DFT matrix (columns = basis vectors)
    Finv = np.fft.ifft(np.eye(N), axis=0)          # inverse DFT

    # Diagonal multipliers for d/dx, d²/dx², d³/dx³
    ik1 = 1j * k
    ik2 = (1j * k) ** 2
    ik3 = (1j * k) ** 3

    D  = np.real(Finv @ np.diag(ik1) @ F)
    D2 = np.real(Finv @ np.diag(ik2) @ F)
    D3 = np.real(Finv @ np.diag(ik3) @ F)

    xi = xi.reshape(-1, 1)
    w_flat = np.full(N, h)                         # uniform quadrature
    w = w_flat.reshape(-1, 1)

    return xi, D, D2, D3, w_flat, w


def _build_full_grid_operators(n_states: int):
    """
    Build Chebyshev operators on the full grid on [0, 1] (including boundaries).
    Grid: xi[0] = 0, xi[-1] = 1. Scale: d/dxi = 2 * d/dz with z in [-1, 1].
    Returns xi, D, D2, D3, D4, w_flat, w for use with Robin/nonlinear BCs.
    """
    # cheb(N) returns N+1 points; we want n_states points so use cheb(n_states - 1)
    N = n_states
    x_nodes, D_z, w_z = cheb(N - 1)
    # x_nodes: [1, ..., -1] (first is right, last is left in [-1,1])
    # Map to [0,1] with xi[0]=0, xi[-1]=1: reverse and scale xi = (z+1)/2
    xi = (x_nodes[::-1] + 1.0) / 2.0
    xi = xi.reshape(-1, 1)
    # Reverse differentiation matrices and scale: d/dxi = 2 * d/dz
    P = np.eye(N)[::-1]  # reversal permutation
    D_z_rev = P @ D_z @ P.T
    D = 2.0 * D_z_rev
    D2_z = D_z @ D_z
    D3_z = D_z @ D2_z
    D4_z = D2_z @ D2_z
    D2 = 4.0 * (P @ D2_z @ P.T)
    D3 = 8.0 * (P @ D3_z @ P.T)
    D4 = 16.0 * (P @ D4_z @ P.T)
    w_flat = (w_z[::-1] / 2.0)
    w = w_flat.reshape(-1, 1)
    return xi, D, D2, D3, D4, w_flat, w


def sample_initial_conditions_fourier(
    xi: np.ndarray,
    n_states: int,
    n: int,
    *,
    modes: int = 5,
    scale: float = 1.0,
    use_sine: bool = True,
    use_cosine: bool = False,
    domain_type: str = "symmetric",
    L: float = None,
    seed: int = None,
    **kwargs,
) -> np.ndarray:
    """
    Unified Fourier-sum sampling of initial conditions (Burgers-style).
    X0 = scale * sum_{k=1}^K [ a_k * sin(...) + b_k * cos(...) ] with
    a_k, b_k = (2*rand-1) / k^0.5.

    Parameters
    ----------
    xi : (n,) array
        Spatial grid.
    n_states, n : int
        State dimension and number of samples.
    modes : int
        Number of Fourier modes K.
    scale : float
        Final multiplier.
    use_sine, use_cosine : bool
        Whether to include sine (a_k) and/or cosine (b_k) terms.
    domain_type : str
        'symmetric' for [-1,1], 'periodic' for [0,L).
    L : float
        Domain length for periodic.
    seed : int, optional
        Random seed.
    **kwargs
        Ignored (for API compatibility).

    Returns
    -------
    X0 : (n_states, n) array
    """
    if seed is not None:
        np.random.seed(seed)
    xi_flat = np.asarray(xi).flatten()
    d = len(xi_flat)
    X0 = np.zeros((d, n), dtype=float)

    if domain_type == "symmetric":
        # xi in [-1, 1]: sin(k*pi*xi), cos(k*pi*xi)
        arg = np.pi * xi_flat
        for k in range(1, modes + 1):
            if use_sine:
                ak = (2.0 * np.random.rand(1, n) - 1.0) / float(k) ** 0.5
                X0 += ak * np.sin(k * arg).reshape(-1, 1)
            if use_cosine:
                bk = (2.0 * np.random.rand(1, n) - 1.0) / float(k) ** 0.5
                X0 += bk * np.cos(k * arg).reshape(-1, 1)
    elif domain_type == "periodic":
        if L is None:
            raise ValueError("L required for periodic domain")
        # xi in [0, L): sin(2*pi*k*xi/L), cos(2*pi*k*xi/L)
        arg = 2.0 * np.pi * xi_flat / L
        for k in range(1, modes + 1):
            if use_sine:
                ak = (2.0 * np.random.rand(1, n) - 1.0) / float(k) ** 0.5
                X0 += ak * np.sin(k * arg).reshape(-1, 1)
            if use_cosine:
                bk = (2.0 * np.random.rand(1, n) - 1.0) / float(k) ** 0.5
                X0 += bk * np.cos(k * arg).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown domain_type: {domain_type!r}")

    return scale * X0


def spatial_control_matrix(
    xi: np.ndarray,
    n_controls: int,
    width: float,
    *,
    domain_type: str = "symmetric",
    x_min: float = -1.0,
    x_max: float = 1.0,
    L: float = None,
) -> np.ndarray:
    """
    Build spatial control matrix B with quadratic bump actuation (Burgers-style).
    Actuators are evenly spaced, each column normalized to unit peak.

    width is the fraction of domain length L: each bump spans L*width, so with
    m controls the total footprint is m * L * width (e.g. 3 controls and
    width=0.2 covers 0.6*L of the domain).

    Each actuator j has shape b_j(x) = (x - left) * (x - right) for x in
    [left, right]; zero elsewhere. Columns are scaled so max(b_j) = 1.

    Parameters
    ----------
    xi : (n,) or (n, 1) array
        Spatial grid points.
    n_controls : int
        Number of actuators.
    width : float
        Fraction of domain length (0, 1]. Each bump has physical width L * width.
    domain_type : str
        'symmetric' for bounded [x_min, x_max], 'periodic' for [0, L).
    x_min, x_max : float
        For domain_type='symmetric', domain bounds.
    L : float
        For domain_type='periodic', domain length.

    Returns
    -------
    B : (n, n_controls) array
    """
    xi_flat = np.asarray(xi).flatten()
    n = len(xi_flat)

    if domain_type == "symmetric":
        domain_L = x_max - x_min
    elif domain_type == "periodic":
        if L is None:
            raise ValueError("L required for periodic domain")
        domain_L = L
    else:
        raise ValueError(f"Unknown domain_type: {domain_type!r}")

    bump_width = domain_L * width
    eps = bump_width / 2.0

    if domain_type == "symmetric":
        centers = np.linspace(x_min + bump_width, x_max - bump_width, n_controls)
    elif domain_type == "periodic":
        centers = np.array([domain_L * j / n_controls for j in range(n_controls)])
    else:
        raise ValueError(f"Unknown domain_type: {domain_type!r}")

    cols = []
    for c in centers:
        left, right = c - eps, c + eps
        mask = (xi_flat >= left) & (xi_flat <= right)
        g = np.zeros(n, dtype=float)
        g[mask] = (xi_flat[mask] - left) * (xi_flat[mask] - right)
        g = -g
        g = np.abs(g)
        cols.append(g.reshape(-1, 1))
    B = np.hstack(cols)

    col_max = B.max(axis=0, keepdims=True)
    col_max[col_max <= 0] = 1.0
    B = B / col_max

    return B


class BaseOCP:
    """Base class for optimal control problems - contains general methods."""
    
    def get_params(self, **params):
        '''
        Function to return a dict of parameters which might be needed by matlab
        scripts.

        Arguments
        ----------
        params : keyword arguments
            Additional parameters to return.

        Returns
        ----------
        params_dict : dict
            Dict of name-value pairs including
            'n_states' : int
            'n_controls' : int
            'X_bar' : (n_states, 1) array
            'U_bar' : (n_controls, 1) array
            'P' : (n_states, n_states) array
            'K' : (n_controls, n_states) array
            'xi' : (n_states, 1) array
            'w' : (n_states, 1) array
            **params
        '''
        params_dict = {
            'n_states': self.n_states,
            'n_controls': self.n_controls,
            'X_bar': self.X_bar,
            'U_bar': self.U_bar,
            'A': self._A,
            'B': self._B,
            'Q': self._Q,
            'R': self._R,
            'P': self.LQR.P,
            'K': self.LQR.K,
            **params
        }

        params_dict['xi'] = self.xi

        params_dict['w'] = self.w
        return params_dict

    def norm(self, X, center_X_bar=True):
        """
        Default norm: weighted distance from reference. Uses self.w (quadrature
        or other weights) and self.X_bar (None = regulate to zero). Subclasses
        without Clenshaw–Curtis or with a different norm should override.
        """
        X = X.reshape(self.n_states, -1)
        ref = np.zeros((self.n_states, 1)) if self.X_bar is None else self.X_bar.reshape(self.n_states, -1)
        if ref.shape[1] == 1 and X.shape[1] != 1:
            ref = np.broadcast_to(ref, X.shape)
        err = X - ref
        if not center_X_bar:
            err = X
        return np.sqrt(np.sum(err**2 * self.w, axis=0))


    def convergence(self, X, U, conv_tol, fp_tol):
        """
        Vectorized convergence test: running cost small AND dynamics small.
        Same criterion as IndirectSolver.check_converged (the BVP gold standard).
        Returns boolean array (one entry per column of X).
        """
        cost_small = self.running_cost(X, U) < conv_tol
        dynamics_small = np.linalg.norm(self.dynamics(X, U), axis=0) < fp_tol
        return cost_small & dynamics_small

    def convergence_torch(self, f, L, conv_tol, fp_tol):
        """
        Torch mirror of convergence for training rollouts.
        f: (B,d) dynamics, L: (B,) running cost. Returns (B,) bool tensor.
        """
        cost_small = L < conv_tol
        dynamics_small = torch.norm(f, dim=1) < fp_tol
        return cost_small & dynamics_small

    def _state_error(self, X):
        """State error from reference (self.X_bar; None = zero). Used in running cost."""
        X = X.reshape(self.n_states, -1)
        ref = np.zeros((self.n_states, 1)) if self.X_bar is None else np.asarray(self.X_bar).reshape(self.n_states, -1)
        if ref.shape[1] == 1 and X.shape[1] != 1:
            ref = np.broadcast_to(ref, X.shape)
        return X - ref

    def running_cost(self, X, U, wX=None):
        '''
        Evaluate the running cost L(X,U) = ||X - X_ref||^2_w + R||U||^2 at one or
        multiple state-control pairs. X_ref is self.X_bar if set, else zero.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        wX : (n_states,) or (n_states, n_points) array, optional
            Precomputed w * X (not used if X_ref is X_bar; pass wX only for legacy).

        Returns
        -------
        L : (1,) or (n_points,) array
            Running cost(s) L(X,U) evaluated at pair(s) (X,U).
        '''
        X_err = self._state_error(X)
        if wX is None:
            if X_err.ndim == 1:
                wX_err = self.w_flat * X_err
            else:
                wX_err = self.w * X_err
        else:
            # Legacy: wX was w*X; convert to w*(X - X_bar) when X_bar is set
            if self.X_bar is None:
                wX_err = wX
            else:
                ref = np.asarray(self.X_bar).reshape(self.n_states, -1)
                if ref.shape[1] == 1 and X_err.shape[1] != 1:
                    ref = np.broadcast_to(ref, X_err.shape)
                wX_err = wX - (self.w * ref if X_err.ndim > 1 else self.w_flat * ref.flatten())
        return np.sum(wX_err * X_err, axis=0) + self.R * np.sum(U**2, axis=0)

    def running_cost_gradient(self, X, U, return_dLdX=True, return_dLdU=True):
        '''
        Evaluate the gradients of the running cost: dL/dX = 2*w*(X - X_ref),
        dL/dU = 2*R*U, with X_ref = self.X_bar if set else zero.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        return_dLdX : bool, default=True
            Set to True to compute the gradient with respect to states, dL/dX.
        return_dLdU : bool, default=True
            Set to True to compute the gradient with respect to controls, dL/dU.

        Returns
        -------
        dLdX : (n_states,) or (n_states, n_points) array
            Gradient dL/dX (X,U) evaluated at pair(s) (X,U).
        dLdU : (n_states,) or (n_states, n_points) array
            Gradient dL/dU (X,U) evaluated at pair(s) (X,U).
        '''
        X_err = self._state_error(X)
        if return_dLdX:
            if X_err.ndim == 1:
                dLdX = 2. * (self.w_flat * X_err)
            else:
                dLdX = 2. * (self.w * X_err)
            if not return_dLdU:
                return dLdX

        if return_dLdU:
            dLdU = 2. * self.R * U
            if not return_dLdX:
                return dLdU

        return dLdX, dLdU

    def U_star(self, X, dVdX):
        '''
        Evaluate the optimal control as a function of state and costate.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).

        Returns
        -------
        U : (n_controls,) or (n_controls, n_points) array
            Optimal control(s) arranged by (dimension, time).
        '''
        U = np.matmul(self.RBT, dVdX)
        return U

    def make_bc(self, X0):
        '''
        Generates a function to evaluate the boundary conditions for a given
        initial condition. Terminal cost is zero so final condition on lambda is
        zero.

        Parameters
        ----------
        X0 : (n_states, 1) array
            Initial condition.

        Returns
        -------
        bc : callable
            Function of X_aug_0 (augmented states at initial time) and X_aug_T
            (augmented states at final time), returning a function which
            evaluates to zero if the boundary conditions are satisfied.
        '''
        X0 = X0.flatten()
        def bc(X_aug_0, X_aug_T):
            return np.concatenate((
                X_aug_0[:self.n_states] - X0, X_aug_T[self.n_states:]
            ))
        return bc

    def Hamiltonian(self, X, U, dVdX):
        '''
        Evaluate the Pontryagin Hamiltonian,
        H(X,U,dVdX) = L(X,U) + <dVdX, F(X,U)>
        where L(X,U) is the running cost, dVdX is the costate or value gradient,
        and F(X,U) is the dynamics. A necessary condition for optimality is that
        H(X,U,dVdX) ~ 0 for the whole trajectory.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Value gradient dV/dX (X,U) evaluated at pair(s) (X,U).

        Returns
        -------
        H : (1,) or (n_points,) array
            Pontryagin Hamiltonian at each point in time.
        '''
        L = self.running_cost(X, U)
        F = self.dynamics(X, U)
        return L + np.sum(dVdX * F, axis=0)

    def compute_cost(self, t, X, U):
        '''Computes the accumulated cost J(t) of a state-control trajectory.'''
        L = self.running_cost(X, U)
        J = cumtrapz(L.flatten(), t)
        return np.concatenate((J, J[-1:]))

    def closed_loop_jacobian(self, X, controller):
        '''
        Evaluate the Jacobian of the closed-loop dynamics at single or multiple
        time instances.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current states.
        controller : object
            Controller instance implementing eval_U and eval_dUdX methods.

        Returns
        -------
        dFdX : (n_states, n_states) or (n_states, n_states, n_points) array
            Closed-loop Jacobian dF/dX + dF/dU * dU/dX.
        '''
        dFdX, dFdU = self.jacobians(X, controller.eval_U(X))
        dUdX = controller.eval_dUdX(X)

        while dFdU.ndim < 3:
            dFdU = dFdU[...,None]
        while dUdX.ndim < 3:
            dUdX = dUdX[...,None]

        dFdX += np.einsum('ijk,jhk->ihk', dFdU, dUdX)

        if X.ndim < 2:
            dFdX = np.squeeze(dFdX)

        return dFdX

    def apply_state_constraints(self, X):
        '''
        Manually update states to satisfy some state constraints. At present
        time, the OCP format only supports constraints which are intrinsic to
        the dynamics (such as quaternions or periodicity), not dynamic
        constraints which need to be satisfied by admissible controls.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states.

        Returns
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states with constrained values.
        '''
        return X

    def constraint_fun(self, X):
        '''
        A (vector-valued) function which is zero when the state constraints are
        satisfied. At present time, the OCP format only supports constraints
        which are intrinsic to the dynamics (such as quaternions or
        periodicity), not dynamic constraints which need to be satisfied by
        admissible controls.

        Arguments
        ----------
        X : (n_states, n_data) or (n_states,) array
            Current states.

        Returns
        ----------
        C : (n_constraints,) or (n_constraints, n_data) array or None
            Algebraic equation such that C(X)=0 means that X satisfies the state
            constraints.
        '''
        return None

    def constraint_jacobian(self, X):
        '''
        Constraint function Jacobian dC/dX of self.constraint_fun. Default
        implementation approximates this with central differences.

        Parameters
        ----------
        X : (n_states,) array
            Current state.

        Returns
        -------
        dCdX : (n_constraints, n_states) array or None
            dC/dX evaluated at the point X, where C(X)=self.constraint_fun(X).
        '''
        C0 = self.constraint_fun(X)
        if C0 is None:
            return None

        return approx_derivative(self.constraint_fun, X, f0=C0)


    def sample_initial_conditions(self, n: int, seed=None, **kwargs):
        """
        Sample n initial conditions. Override in each problem; default raises
        NotImplementedError. Returns (n_states, n) array. kwargs (e.g. K for
        Fourier modes) are passed through for problem-specific options.
        """
        raise NotImplementedError("Subclass must implement sample_initial_conditions")

    def make_integration_events(self, x_max=4.0):
        import numpy as np

        def explosion_event(t, X):
            # X comes in as (n_states,) for solve_ivp with vectorized=False
            return np.max(np.abs(X)) - x_max

        explosion_event.terminal = True
        explosion_event.direction = 1
        return [explosion_event]

    def physics_module(self):
        """Return a torch.nn.Module that wraps this OCP's dynamics and cost for autodiff."""
        if torch is None:
            raise ImportError("PyTorch is required for physics_module()")
        return BasePhysics(self)


# ---------------------------------------------------------------------------
# Torch physics wrapper: single source of truth from OCP (NumPy) methods
# ---------------------------------------------------------------------------

if torch is not None:

    class _OCPDynamicsFunc(torch.autograd.Function):
        """Forward: ocp.dynamics; backward: ocp.jacobians."""

        @staticmethod
        def forward(ctx, x, u, ocp):
            # x (B, n), u (B, m) -> X (n, B), U (m, B)
            X = x.detach().cpu().numpy().T
            U = u.detach().cpu().numpy().T
            F = ocp.dynamics(X, U)
            if F.ndim == 1:
                F = F.reshape(-1, X.shape[1])
            ctx.ocp = ocp
            ctx.save_for_backward(x, u)
            out = torch.from_numpy(F.T.copy()).to(device=x.device, dtype=x.dtype)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            ocp = ctx.ocp
            x, u = ctx.saved_tensors
            X = x.cpu().numpy().T   # (n, B)
            U = u.cpu().numpy().T   # (m, B)
            dFdX, dFdU = ocp.jacobians(X, U)  # (n, n, B), (n, m, B)
            go = grad_output.cpu().numpy().T  # (n, B)
            # grad_x[b,i] = sum_j dFdX[j,i,b]*go[j,b]  -> (B, n)
            grad_x = np.einsum("jib,jb->bi", dFdX, go)
            # grad_u[b,k] = sum_i dFdU[i,k,b]*go[i,b]  -> (B, m)
            grad_u = np.einsum("ikb,ib->bk", dFdU, go)
            return (
                torch.from_numpy(grad_x).to(x.device, x.dtype),
                torch.from_numpy(grad_u).to(u.device, u.dtype),
                None,
            )

    class _OCPRunningCostFunc(torch.autograd.Function):
        """Forward: ocp.running_cost; backward: ocp.running_cost_gradient."""

        @staticmethod
        def forward(ctx, x, u, ocp):
            X = x.detach().cpu().numpy().T
            U = u.detach().cpu().numpy().T
            L = ocp.running_cost(X, U)
            if np.isscalar(L):
                L = np.full(X.shape[1], L, dtype=X.dtype)
            ctx.ocp = ocp
            ctx.save_for_backward(x, u)
            out = torch.from_numpy(np.asarray(L).ravel().copy()).to(device=x.device, dtype=x.dtype)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            ocp = ctx.ocp
            x, u = ctx.saved_tensors
            X = x.cpu().numpy().T
            U = u.cpu().numpy().T
            dLdX, dLdU = ocp.running_cost_gradient(X, U)
            go = grad_output.cpu().numpy().ravel()  # (B,)
            grad_x = (go[:, None] * dLdX.T).astype(X.dtype)
            grad_u = (go[:, None] * dLdU.T).astype(U.dtype)
            return (
                torch.from_numpy(grad_x).to(x.device, x.dtype),
                torch.from_numpy(grad_u).to(u.device, u.dtype),
                None,
            )

    class BasePhysics(torch.nn.Module):
        """
        Torch wrapper for any BaseOCP. dynamics and running_cost delegate to the
        OCP's NumPy methods (with autograd via jacobians / running_cost_gradient).
        get_control uses OCP's B and R (differentiable in torch).
        """

        def __init__(self, ocp, u_clamp=10.0):
            super().__init__()
            self._ocp = ocp
            self.u_clamp = float(u_clamp)
            self.register_buffer("B", torch.from_numpy(np.asarray(ocp.B)).float())
            R = getattr(ocp, "R", 0.5)
            self.R_val = float(R) if np.isscalar(R) else float(np.asarray(R).flat[0])

        def get_control(self, dVdX: torch.Tensor) -> torch.Tensor:
            # u* = -(1/(2R)) B^T dVdX  ->  (B, m) from (B, n) @ (n, m)
            u = -(0.5 / self.R_val) * (dVdX @ self.B)
            return torch.clamp(u, -self.u_clamp, self.u_clamp)

        def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return _OCPDynamicsFunc.apply(x, u, self._ocp)

        def running_cost(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return _OCPRunningCostFunc.apply(x, u, self._ocp)


def attach_ocp(config, ocp_class, defaults, overrides=None):
    """Shared setup: create OCP, copy convenience attributes to config."""
    overrides = overrides or {}
    for k, v in defaults.items():
        setattr(config, k, overrides.get(k, v))
    config.ocp = ocp_class(config)
    config.B = config.ocp.B
    config.q = config.ocp.R
    config.x_f = config.ocp.X_bar.flatten()
    config.xi = config.ocp.xi
    config.w = config.ocp.w
    config.norm = config.ocp.norm
    config.dynamics = config.ocp.dynamics
    config.running_cost = config.ocp.running_cost
    config.running_cost_gradient = config.ocp.running_cost_gradient
    config.f_controlled = lambda t, x, u, cfg: cfg.ocp.dynamics(x, u)
    config.jacobian_f = lambda x, cfg: cfg.ocp.jacobians(x, np.zeros((cfg.n_controls,)))[0]

    # Derived: ic_basis for display (sine/cosine/both)
    s = getattr(config, "ic_use_sine", False)
    c = getattr(config, "ic_use_cosine", False)
    config.ic_basis = "both" if (s and c) else ("sine" if s else ("cosine" if c else "none"))
