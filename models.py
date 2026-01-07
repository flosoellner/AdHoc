import numpy as np
from scipy.linalg import solve_continuous_are as care


class LQR:
    '''
    Linear quadratic regulator (LQR) controller.
    '''
    def __init__(self, X_bar, U_bar, A, B, Q, R, P=None):
        self.X_bar = np.reshape(X_bar, (-1, 1))
        self.U_bar = np.reshape(U_bar, (-1, 1))
        self.n_states = self.X_bar.shape[0]
        self.n_controls = self.U_bar.shape[0]

        # Make Riccati matrix and LQR control gain matrix
        if P is not None:
            self.P = np.asarray(P)
        else:
            self.P = care(A, B, Q, R)
        self.RB = np.linalg.solve(R, np.transpose(B))
        self.K = np.matmul(self.RB, self.P)

    def eval_V(self, X):
        '''Value function V(X) = (X - X_bar)^T P (X - X_bar)'''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar
        XPX = X_err * np.matmul(self.P, X_err)
        XPX = np.sum(XPX, axis=0, keepdims=True)
        return XPX.flatten() if X.ndim < 2 else XPX

    def eval_dVdX(self, X):
        '''Value gradient dV/dX(X) = 2*P*(X - X_bar)'''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar
        PX = 2. * np.matmul(self.P, X_err)
        return PX.reshape(X.shape)

    def eval_U(self, X):
        '''Control U(X) = U_bar - K*(X - X_bar)'''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar
        U = self.U_bar - np.matmul(self.K, X_err)
        return U.flatten() if X.ndim < 2 else U

    def control(self, X):
        '''Control U(X) = U_bar - K*(X - X_bar)'''
        return self.eval_U(X)

    def value_gradient(self, X):
        '''Value gradient dV/dX(X) = 2*P*(X - X_bar)'''
        return self.eval_dVdX(X)

    def Vx(self, X):
        '''Value gradient dV/dX(X) = 2*P*(X - X_bar)'''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar
        PX = 2. * np.matmul(self.P, X_err)
        return PX.reshape(X.shape)

    def eval_dUdX(self, X):
        '''Control Jacobian dU/dX = -K'''
        if X.ndim < 2:
            return -self.K
        dUdX = np.expand_dims(-self.K, -1)
        return np.tile(dUdX, (1, 1, X.shape[1]))

    def bvp_guess(self, X):
        '''Returns (V, dVdX, U) for BVP initial guess'''
        X_err = X.reshape(X.shape[0], -1) - self.X_bar
        PX = np.matmul(self.P, X_err)
        XPX = np.sum(X_err * PX, axis=0, keepdims=True)
        U = self.U_bar - np.matmul(self.RB, PX)
        PX = 2. * PX.reshape(X.shape)
        if X.ndim < 2:
            return XPX.flatten(), PX, U.flatten()
        return XPX, PX, U


class LQRController:
    '''
    Wrapper for LQR to work with ProblemConfig interface.
    '''
    def __init__(self, config):
        if hasattr(config, 'ocp') and hasattr(config.ocp, 'LQR'):
            self.lqr = config.ocp.LQR
        elif hasattr(config, 'LQR'):
            self.lqr = config.LQR
        else:
            raise AttributeError("config must have LQR attribute")
        self.config = config

    def control(self, x):
        '''Control input U(x)'''
        return self.lqr.control(x)

    def value_gradient(self, x):
        '''Value gradient dV/dX(x)'''
        return self.lqr.value_gradient(x)

    def Vx(self, x):
        '''Value gradient dV/dX(x)'''
        return self.lqr.Vx(x)
