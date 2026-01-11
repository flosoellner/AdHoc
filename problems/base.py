import numpy as np
try:
    from scipy.integrate import cumtrapz
except:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.optimize._numdiff import approx_derivative
from scipy import sparse


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

def find_fixed_point(OCP, controller, tol, X0=None, verbose=True):
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
    if np.abs(max_eig.real) < tol**2:
        Jac0 = np.squeeze(OCP.closed_loop_jacobian(OCP.X_bar, OCP.LQR))
        eigs0 = np.linalg.eigvals(Jac0)
        idx = np.argsort(eigs0.real)
        eigs0 = eigs0[idx].reshape(-1,1)
        max_eig0 = np.squeeze(eigs0[-1])

        i = 2
        while all([
                i <= OCP.n_states,
                np.abs(max_eig.real) < tol**2,
                np.abs(max_eig0.real) < tol**2
            ]):
            max_eig = np.squeeze(eigs[OCP.n_states - i])
            max_eig0 = np.squeeze(eigs0[OCP.n_states - i])
            i += 1

    if verbose:
        s = '||actual - desired_equilibrium|| = {norm:1.2e}'
        print(s.format(norm=X_star_err))
        if np.max(np.abs(F_star)) > tol:
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
        '''
        Calculate the distance of a batch of spatial points from X_bar or zero.
        Uses the Clenshaw Curtis quadrature weights to compute a weighted norm.

        Arguments
        ----------
        X : (n_states, n_data) array
            Points to compute distances for
        center_X_bar : not used
            For API consistency only

        Returns
        ----------
        X_norm : (n_data,) array
            Norm for each point in X
        '''
        X = X.reshape(self.n_states, -1)

        return np.sqrt(np.sum(X**2 * self.w, axis=0))


    def running_cost(self, X, U, wX=None):
        '''
        Evaluate the running cost L(X,U) at one or multiple state-control pairs.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        U : (n_controls,) or (n_controls, n_points) array
            Control(s) arranged by (dimension, time).
        wX : (n_states,) or (n_states, n_points) array, optional
            States(s) multiplied by the Chebyshev quadrature weights.

        Returns
        -------
        L : (1,) or (n_points,) array
            Running cost(s) L(X,U) evaluated at pair(s) (X,U).
        '''
        if wX is None:
            if X.ndim == 1:
                wX = self.w_flat * X 
            else:
                wX = self.w * X 

        return np.sum(wX * X, axis=0) + self.R * np.sum(U**2, axis=0)

    def running_cost_gradient(self, X, U, return_dLdX=True, return_dLdU=True):
        '''
        Evaluate the gradients of the running cost, dL/dX (X,U) and dL/dU (X,U),
        at one or multiple state-control pairs.

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
        if return_dLdX:
            if X.ndim == 1:
                dLdX = 2. * (self.w_flat * X) 
            else:
                dLdX = 2. * (self.w * X)
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

    def jac_U_star(self, X, dVdX, U0=None):
        '''
        Evaluate the Jacobian of the optimal control with respect to the state,
        leaving the costate fixed.

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            State(s) arranged by (dimension, time).
        dVdX : (n_states,) or (n_states, n_points) array
            Costate(s) arranged by (dimension, time).
        U0 : ignored
            For API consistency only

        Returns
        -------
        U : (n_controls,) or (n_controls, n_points) array
            Optimal control(s) arranged by (dimension, time).
        '''
        dVdX = dVdX.reshape(self.n_states, -1)
        return np.zeros((self.n_controls, self.n_states, dVdX.shape[-1]))

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

    def make_integration_events(self, x_max=4.0):
        import numpy as np

        def explosion_event(t, X):
            # X comes in as (n_states,) for solve_ivp with vectorized=False
            return np.max(np.abs(X)) - x_max

        explosion_event.terminal = True
        explosion_event.direction = 1
        return [explosion_event]
