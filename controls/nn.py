import numpy as np
import torch
import torch.nn as nn
from problems.base import u_analytic


def make_mlp(d_in, d_out, hidden=64, depth=4, act=nn.Tanh):
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


class ResBlock(nn.Module):
    def __init__(self, width, act=nn.Tanh, alpha=0.1, use_ln=False):
        super().__init__()
        self.alpha = float(alpha)
        self.ln = nn.LayerNorm(width) if use_ln else nn.Identity()
        self.fc1 = nn.Linear(width, width)
        self.act = act()
        self.fc2 = nn.Linear(width, width)

        # optional but nice: start near-identity
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.zero_()

    def forward(self, x):
        y = self.ln(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        return x + self.alpha * y


class ResMLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=64, depth=4, act=nn.Tanh, alpha=0.1, use_ln=False):
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.act = act()
        self.blocks = nn.ModuleList([ResBlock(hidden, act=act, alpha=alpha, use_ln=use_ln) for _ in range(depth)])
        self.out = nn.Linear(hidden, d_out)

    def forward(self, x):
        x = self.act(self.inp(x))
        for b in self.blocks:
            x = b(x)
        return self.out(x)


class GradNet(nn.Module, TorchWrapperMixin):
    def __init__(self, config, hidden=64, depth=2, act=nn.Tanh, use_lqr=True, *, residual=True, res_alpha=0.1, res_ln=False):
        super().__init__()
        self.config = config
        self.use_lqr = use_lqr
        d = config.n_states

        if residual:
            self.net = ResMLP(d, d, hidden=hidden, depth=depth, act=act, alpha=res_alpha, use_ln=res_ln)
            last = self.net.out
        else:
            self.net = make_mlp(d, d, hidden=hidden, depth=depth, act=act)
            last = self.net[-1]

        # silent start (keeps your "doesn't wreck LQR at epoch 0" behavior)
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()

    def forward(self, x):
        g_res = self.net(x)
        g0 = self.net(x.new_zeros((1, x.shape[1])))
        g_res = g_res - g0

        if not self.use_lqr:
            return g_res

        X_np = x.detach().cpu().numpy().T
        g_lqr_np = self.config.ocp.LQR.eval_dVdX(X_np)
        g_lqr = torch.tensor(g_lqr_np.T, dtype=x.dtype, device=x.device)
        return g_lqr + g_res


class Control(nn.Module, TorchWrapperMixin):
    """
    Uses analytic control from value gradient.
    Optionally adds an LQR baseline and/or uses a residual value gradient.
    """
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
            g = grad_from_value_autograd(self.value_net, x)
        else:
            raise ValueError(...)

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


def make_controller(config, kind="lqr", *, grad_net=None, value_net=None):
    """
    Returns an object with .eval_U(X) usable by simulation/data generation.
    """
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


