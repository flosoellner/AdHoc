import numpy as np

from .base import BaseOCP, cheb
from models import LQR

class BurgersOCP(BaseOCP):
    """Burgers equation optimal control problem."""
    
    def __init__(self, config):

        n_states = config.n_states
        n_controls = config.n_controls
        # Burgers-specific parameters
        self.nu = 0.02
        self.gamma = 0.1
        self.R = 0.5
        kappa = 25.

        # Chebyshev nodes, differentiation matrices, and Clenshaw-Curtis weights
        self.xi, self.D, self.w_flat = cheb(n_states + 1)
        self.D2 = np.matmul(self.D, self.D)

        # Truncate system to account for zero boundary conditions
        self.xi = self.xi[1:-1].reshape(-1,1)
        self.w_flat = self.w_flat[1:-1]
        self.w = self.w_flat.reshape(-1,1)
        self.D = self.D[1:-1, 1:-1]
        self.D2 = self.D2[1:-1, 1:-1]

        # Control multiplier - Burgers-specific spatial locations
        B = np.hstack((
            (-4/5 <= self.xi) & (self.xi <= -2/5),
            (2/5 <= self.xi) & (self.xi <= 4/5)
        ))
        B = -kappa * B * np.hstack((
            (self.xi + 4/5)*(self.xi + 2/5),
            (self.xi - 2/5)*(self.xi - 4/5)
        ))
        B = np.abs(B)

        # Forcing term coefficient - Burgers-specific
        self.alpha = np.abs(self.xi) <= 1/5
        self.alpha = - kappa * self.alpha * (self.xi + 1/5)*(self.xi - 1/5)
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


    def dynamics(self, X, U):
        '''
        Burgers equation dynamics: dXdt = -0.5*D*X² + nu*D2*X + X*alpha*exp(-gamma*X) + B*U

        Parameters
        ----------
        X : (n_states,) or (n_states, n_points) array
            Current state.
        U : (n_controls,) or (n_controls, n_points)  array
            Feedback control U=U(X).

        Returns
        -------
        dXdt : (n_states,) or (n_states, n_points) array
            Dynamics dXdt = F(X,U).
        '''
        flat_out = X.ndim < 2
        X = X.reshape(self.n_states, -1)
        U = U.reshape(self.n_controls, -1)

        dXdt = (
            - 0.5*np.matmul(self.D, X**2)
            + np.matmul(self.nu*self.D2, X)
            + X * self.alpha * np.exp(-self.gamma * X)
            + np.matmul(self.B, U)
        )

        if flat_out:
            dXdt = dXdt.flatten()

        return dXdt

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

        gamma_X = -self.gamma * X
        gamma_X = (1. + gamma_X) * self.alpha * np.exp(gamma_X)

        dFdX = (
            - X * np.expand_dims(self.D, -1)
            + np.expand_dims(self.nu * self.D2, -1)
        )

        diag_idx = np.diag_indices(self.n_states)
        for k in range(X.shape[1]):
            dFdX[diag_idx[0],diag_idx[1],k] += gamma_X[:,k]

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
        aeX = self.alpha * np.exp(-self.gamma * X)

        dXdt = (
            - 0.5*np.matmul(self.D, X**2)
            + np.matmul(self.nu*self.D2, X)
            + X * aeX
            + np.matmul(self.B, U)
        )

        dAdt = (
            - 2.*wX
            + X * np.matmul(self.D.T, A)
            - np.matmul(self.nu * self.D2.T, A)
            - aeX * (1. - self.gamma*X) * A
        )

        L = np.atleast_2d(self.running_cost(X, U, wX))

        return np.vstack((dXdt, dAdt, -L))




