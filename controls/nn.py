import numpy as np
import torch
import torch.nn as nn
from problems.base import u_analytic


def make_mlp(d_in, d_out, hidden=None, depth=None, act=nn.Tanh):
    layers = [nn.Linear(d_in, hidden), act()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), act()]
    layers += [nn.Linear(hidden, d_out)]
    return nn.Sequential(*layers)



class TorchWrapperMixin:
    """Numpy <-> torch convenience for eval_* methods."""
    def _to_tensor(self, X):
        X = np.asarray(X)
        single = (X.ndim == 1)
        if single:
            x = torch.tensor(X, dtype=torch.float32).unsqueeze(0)   # (1,d)
        else:
            x = torch.tensor(X.T, dtype=torch.float32)              # (N,d)
        return x, single

    def _to_numpy(self, Y, single):
        if single:
            return Y.squeeze(0).detach().cpu().numpy()
        return Y.detach().cpu().numpy().T



class GradNet(nn.Module, TorchWrapperMixin):
    def __init__(self, config, hidden=None, depth=2, act=nn.Tanh, use_lqr=True):
        super().__init__()
        self.config = config
        self.use_lqr = use_lqr
        d = config.n_states
        if hidden is None:
            hidden = 4 * d

        self.net = make_mlp(d, d, hidden=hidden, depth=depth, act=act)
        last = self.net[-1]

        # silent start (keeps your "doesn't wreck LQR at epoch 0" behavior)
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()

    def forward(self, x):
        g_res = self.net(x)
        # Evaluate at X_bar to ensure network output is zero at target
        X_bar_torch = torch.tensor(self.config.ocp.X_bar, dtype=x.dtype, device=x.device).T
        if X_bar_torch.ndim == 1:
            X_bar_torch = X_bar_torch.unsqueeze(0)
        # Expand to match batch dimension
        X_bar_batch = X_bar_torch.expand(1, x.shape[1])
        g0 = self.net(X_bar_batch)
        g_res = g_res - g0

        if not self.use_lqr:
            return g_res

        X_np = x.detach().cpu().numpy().T
        g_lqr_np = self.config.ocp.LQR.eval_dVdX(X_np)
        g_lqr = torch.tensor(g_lqr_np.T, dtype=x.dtype, device=x.device)
        return g_lqr + g_res



class Control(nn.Module, TorchWrapperMixin):
    """Control from value gradient, optionally with LQR baseline."""
    def __init__(self, config, value_net=None, grad_net=None, use_autograd=False):
        super().__init__()
        self.config = config
        self.value_net = value_net
        self.grad_net = grad_net
        self.use_autograd = use_autograd
    def forward(self, x):
        if self.grad_net is not None:
            g = self.grad_net(x)                 # (N,d) torch
        elif self.value_net is not None and self.use_autograd:
            raise NotImplementedError("autograd value-net path not implemented")
        else:
            raise ValueError("Control requires grad_net or value_net")

        # convert to numpy for u_analytic (expects (d,N))
        g_np = g.detach().cpu().numpy().T        # (d,N)
        u_np = u_analytic(g_np, self.config)     # (m,N) numpy

        u = torch.tensor(u_np.T, dtype=torch.float32)  # (N,m)
        return u

    @torch.no_grad()
    def eval_U(self, X):
        x, single = self._to_tensor(X)
        U = self.forward(x)
        return self._to_numpy(U, single)

    def bvp_guess(self, X):
        """Return (V, dVdX, U) for BVP initial guess."""
        # Use NN for dVdX and U
        x, single = self._to_tensor(X)
        with torch.no_grad():
            dVdX_torch = self.grad_net(x)  # (N, d) tensor
        dVdX = self._to_numpy(dVdX_torch, single)  # convert back to numpy
        
        U = self.eval_U(X)  # already implemented
        
        # Compute V: use LQR if grad_net uses LQR, otherwise compute from dVdX
        if self.grad_net is not None and hasattr(self.grad_net, 'use_lqr') and not self.grad_net.use_lqr:
            # Pure GradNet: compute V from dVdX by integration (trapezoidal rule)
            X_bar = self.config.ocp.X_bar
            X_bar_torch = torch.tensor(X_bar, dtype=x.dtype, device=x.device)
            if X_bar_torch.ndim == 1:
                X_bar_torch = X_bar_torch.unsqueeze(0)
            with torch.no_grad():
                dVdX_bar_torch = self.grad_net(X_bar_torch)  # (1, d) tensor
            dVdX_bar = self._to_numpy(dVdX_bar_torch, single=True)  # (d,) - single point
            
            # Compute V using trapezoidal rule: V(X) ≈ 0.5 * (dVdX_bar + dVdX) · (X - X_bar)
            if X.ndim < 2:
                dVdX_bar = dVdX_bar.flatten()
                dVdX_flat = dVdX.flatten()
                X_flat = X.flatten()
                X_err = X_flat - X_bar.flatten()
                dVdX_avg = 0.5 * (dVdX_bar + dVdX_flat)
                V = np.dot(dVdX_avg, X_err)
            else:
                dVdX_bar = dVdX_bar.reshape(-1, 1)
                X_err = X - X_bar
                dVdX_avg = 0.5 * (dVdX_bar + dVdX)
                V = np.sum(dVdX_avg * X_err, axis=0)
            if X.ndim < 2:
                V = float(V) if not np.isscalar(V) else V
                dVdX = dVdX.flatten()
                U = U.flatten()
        else:
            V, _, _ = self.config.ocp.LQR.bvp_guess(X)
            if X.ndim < 2:
                V = V.flatten()[0] if not np.isscalar(V) else V
                dVdX = dVdX.flatten()
                U = U.flatten()
        
        return V, dVdX, U


def make_controller(config, kind="lqr", *, grad_net=None, value_net=None):
    """Return controller with .eval_U(X) for simulation/data generation."""
    kind = kind.lower()

    if kind == "lqr":
        return config.ocp.LQR  # already has eval_U and bvp_guess

    if kind == "nn_grad":
        if grad_net is None:
            raise ValueError("nn_grad controller needs grad_net")
        return Control(config, grad_net=grad_net, use_autograd=False)
    if kind == "nn_value":
        if value_net is None:
            raise ValueError("nn_value controller needs value_net")
        return Control(config, value_net=value_net, use_autograd=True)

    raise ValueError(f"Unknown controller kind: {kind}")


