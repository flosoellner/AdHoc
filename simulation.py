
import warnings
import numpy as np

from scipy.integrate._ivp.ivp import METHODS, MESSAGES
from scipy.integrate._ivp.ivp import OdeResult, OdeSolution, OdeSolver
from scipy.integrate import solve_bvp
import sampling


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _is_finite(*arrays):
    """Return True if every element in all arrays is finite."""
    return all(np.isfinite(a).all() for a in arrays)


# --------------------------------------------------------------------------- #
# IVP solver (modified scipy — events without root-finding)
# --------------------------------------------------------------------------- #

def solve_ivp(
        fun, t_span, y0, method='RK45', t_eval=None, dense_output=False,
        events=None, vectorized=False, args=None, **options
    ):
    """Modified scipy.integrate.solve_ivp: checks events only after each solver
    step (no root-finding). Event times are less accurate but computation is
    faster and more robust. See scipy.integrate.solve_ivp for full docs."""
    t0, tf = map(float, t_span)

    if args is not None:
        # Wrap the user's fun (and jac, if given) in lambdas to hide the
        # additional parameters.  Pass in the original fun as a keyword
        # argument to keep it in the scope of the lambda.
        fun = lambda t, x, fun=fun: fun(t, x, *args)
        jac = options.get('jac')
        if callable(jac):
            options['jac'] = lambda t, x: jac(t, x, *args)

    if t_eval is not None:
        t_eval = np.asarray(t_eval)
        if t_eval.ndim != 1:
            raise ValueError("`t_eval` must be 1-dimensional.")

        if np.any(t_eval < min(t0, tf)) or np.any(t_eval > max(t0, tf)):
            raise ValueError("Values in `t_eval` are not within `t_span`.")

        d = np.diff(t_eval)
        if tf > t0 and np.any(d <= 0) or tf < t0 and np.any(d >= 0):
            raise ValueError("Values in `t_eval` are not properly sorted.")

        if tf > t0:
            t_eval_i = 0
        else:
            # Make order of t_eval decreasing to use np.searchsorted.
            t_eval = t_eval[::-1]
            # This will be an upper bound for slices.
            t_eval_i = t_eval.shape[0]

    if method in METHODS:
        method = METHODS[method]

    with warnings.catch_warnings():
        # Silence warning about unused options
        warnings.filterwarnings(
            'ignore', message='The following arguments have no effect'
        )
        solver = method(fun, t0, y0, tf, vectorized=vectorized, **options)

    if t_eval is None:
        ts = [t0]
        ys = [y0]
    elif t_eval is not None and dense_output:
        ts = []
        ti = [t0]
        ys = []
    else:
        ts = []
        ys = []

    interpolants = []


  

    status = None
    while status is None:
        message = solver.step()

        if solver.status == 'finished':
            status = 0
        elif solver.status == 'failed':
            status = -1
            break

        t_old = solver.t_old
        t = solver.t
        y = solver.y

        if dense_output:
            sol = solver.dense_output()
            interpolants.append(sol)
        else:
            sol = None


        if t_eval is None:
            ts.append(t)
            ys.append(y)
        else:
            # The value in t_eval equal to t will be included.
            if solver.direction > 0:
                t_eval_i_new = np.searchsorted(t_eval, t, side='right')
                t_eval_step = t_eval[t_eval_i:t_eval_i_new]
            else:
                t_eval_i_new = np.searchsorted(t_eval, t, side='left')
                # It has to be done with two slice operations, because
                # you can't slice to 0th element inclusive using backward
                # slicing.
                t_eval_step = t_eval[t_eval_i_new:t_eval_i][::-1]

            if t_eval_step.size > 0:
                if sol is None:
                    sol = solver.dense_output()
                ts.append(t_eval_step)
                ys.append(sol(t_eval_step))
                t_eval_i = t_eval_i_new

        if t_eval is not None and dense_output:
            ti.append(t)

    message = MESSAGES.get(status, message)



    if t_eval is None:
        ts = np.array(ts)
        ys = np.vstack(ys).T
    elif ts:
        ts = np.hstack(ts)
        ys = np.hstack(ys)

    if dense_output:
        if t_eval is None:
            sol = OdeSolution(ts, interpolants)
        else:
            sol = OdeSolution(ti, interpolants)
    else:
        sol = None

    return OdeResult(t=ts, y=ys, sol=sol, t_events=0, y_events=0,
                     nfev=solver.nfev, njev=solver.njev, nlu=solver.nlu,
                     status=status, message=message, success=status >= 0)



def sim_closed_loop(
        dynamics, jacobian, controller, tspan, X0, t_eval=None, events=None,
        solver='LSODA', atol=1e-06, rtol=1e-03, sigma=0, dt_sde=None,
        converged_fn=None, check_every=1
    ):
    '''
    Simulate the closed-loop system for a fixed time interval.

    Parameters
    ----------
        dynamics, jacobian, controller: OCP dynamics and controller
        tspan: integration time, list of two floats
        X0: initial condition, (n,) numpy array
        t_eval: optional (Nt,) numpy array of time instances to evaluate solution at
        events: optional, passed to solve_ivp (ignored when sigma > 0)
        solver, atol, rtol: ODE solver options (used only when sigma <= 0)
        sigma : float, default=0
            Additive noise strength. If sigma > 0, use Euler-Maruyama
            dX = f(X,u) dt + sigma sqrt(dt) dW instead of deterministic ODE.
        dt_sde : float, optional
            Time step for Euler-Maruyama when sigma > 0. If None, use 1e-3.
        converged_fn : callable(x, U) -> bool, optional
            Early-stop callback checked every check_every steps (SDE only).
        check_every : int, default=50
            How often to call converged_fn during SDE integration.

    Returns
    -------
        t: time vector, (Nt,) numpy array
        X: state time series, (n,Nt) numpy array
        status: 0 success, 1 terminal event / explosion, -1 failure
    '''
    if sigma is not None and sigma > 0:
        from scipy.optimize import fsolve
        dt = dt_sde if dt_sde is not None and dt_sde > 0 else 1e-3
        t0, t1 = tspan[0], tspan[1]
        n_steps = max(1, int(np.ceil((t1 - t0) / dt)))
        dt = (t1 - t0) / n_steps
        t = np.linspace(t0, t1, n_steps + 1)
        x = np.asarray(X0).flatten().astype(float)
        n = x.size
        noise = sigma * np.sqrt(dt)
        X_arr = np.empty((n, n_steps + 1))
        X_arr[:, 0] = x
        last_k = 0
        for k in range(n_steps):
            dW = np.random.standard_normal(n)
            b = x + noise * dW

            def residual(z):
                z = np.asarray(z).ravel()
                U = controller.eval_U(z)
                f = np.asarray(dynamics(z, U)).ravel()
                return z - dt * f - b

            try:
                x, info, ier, msg = fsolve(residual, x, full_output=True)
                x = np.asarray(x).ravel()
                if ier != 1 or not np.isfinite(x).all():
                    X_arr[:, k+1] = x
                    return t[:k+2], X_arr[:, :k+2], -1
            except (ValueError, RuntimeError):
                return t[:k+1], X_arr[:, :k+1], -1

            X_arr[:, k+1] = x
            last_k = k + 1
            U = controller.eval_U(x)
            if converged_fn is not None and (k + 1) % check_every == 0:
                if converged_fn(x, U):
                    return t[:last_k+1], X_arr[:, :last_k+1], 0
        return t[:last_k+1], X_arr[:, :last_k+1], 0

    # Deterministic path (unchanged)
    def dynamics_wrapper(t, X):
        U = controller.eval_U(X)
        return dynamics(X, U)

    ode_sol = solve_ivp(
        dynamics_wrapper, tspan, X0, t_eval=t_eval, jac=None,
        events=events, vectorized=True, method=solver, rtol=rtol, atol=atol
    )

    return ode_sol.t, ode_sol.y, ode_sol.status

def sim_to_converge(
        dynamics, jacobian, controller, X0, config, events=None, sigma=0, dt_sde=None
    ):
    '''
    Simulate the closed-loop system until reach t_max or the dX/dt = 0.

    Parameters
    ----------
        dynamics, jacobian, controller: OCP dynamics and controller
        config: a configuration dict defined in problem_def.py
        X0: initial condition, (n,) numpy array
        events: optional
        sigma : float, default=0
            Additive noise strength for SDE (Euler-Maruyama). 0 = deterministic.
        dt_sde : float, optional
            Time step for Euler-Maruyama when sigma > 0.

    Returns
    -------
        t: time vector, (Nt,) numpy array
        X: state time series, (n,Nt) numpy array
        converged: whether or not equilibrium was reached, bool
    '''

    t = np.zeros(1)
    X = X0.reshape(-1,1)

    converged = False

    # Solves over an extended time interval if needed to make ||f(x,u)|| -> 0
    while not converged and t[-1] < config.t1_max:
        X_current = X[:,-1].flatten()
        if not _is_finite(X_current):
            break
        
        t1 = np.maximum(config.t1_initial, t[-1] * config.t1_scale)
        
        def _conv_check(x, U):
            return bool(config.ocp.convergence(
                x.reshape(-1, 1), np.asarray(U).reshape(-1, 1),
                config.conv_tol, config.fp_tol
            ))

        try:
            t_new, X_new, status = sim_closed_loop(
                dynamics,
                jacobian,
                controller,
                [t[-1], t1],
                X_current,
                events=events,
                solver='LSODA',
                atol=1e-06,
                rtol=1e-03,
                sigma=sigma,
                dt_sde=dt_sde,
                converged_fn=_conv_check if (sigma and sigma > 0) else None,
            )
        except (ValueError, RuntimeError) as e:
            # Simulation failed (likely non-finite state)
            break

        if not _is_finite(X_new):
            break

        t = np.concatenate((t, t_new[1:]))
        X = np.hstack((X, X_new[:,1:]))

        if status == 1:
            break

        if not _is_finite(X[:,-1]):
            break

        try:
            U = controller.eval_U(X[:,-1])
            if not _is_finite(U):
                break
            converged = bool(config.ocp.convergence(X[:,-1:], U, config.conv_tol, config.fp_tol))
        except (ValueError, RuntimeError):
            break

    return t, X, converged

from tqdm import tqdm


def monte_carlo(
        OCP, config, controller,
        X0_distance=None, random_seed=None, X0_pool=None,
        n_MC=None, dist=None, K=None,
        solve_open_loop=False, verbose=0, suppress_warnings=True,
        sigma=0, dt_sde=None
    ):
    '''
    Parameters
    ----------
    OCP : instance of QRnet.problem_template.TemplateOCP
    config : instance of QRnet.problem_template.MakeConfig
    controller : controller instance OR list of (name, controller) tuples
        If list, evaluates all controllers with the same initial conditions
    X0_distance : float, optional
        Norm to use for random initial conditions (deprecated, use dist instead)
    random_seed : int, optional
        Seed for numpy random number generator
    X0_pool : array, optional
        Pre-computed initial conditions (n_states, n_MC). If provided, n_MC is inferred from shape.
    n_MC : int, optional
        Number of Monte Carlo samples. Must be set by the experiment if X0_pool is not provided.
    dist : float, optional
        Distance/norm to use for random initial conditions
    K : int, optional
        Passed to sample_conditions for problems that use it (e.g. Fourier modes).
    solve_open_loop : bool, default=False
        Set to True to solve the open loop OCP for each initial condition.
    verbose : int, default=0
        Verbosity level
    suppress_warnings : bool, default=True
        If True, treat numpy warnings as OCP failures.
    sigma : float, default=0
        Additive noise strength for SDE (Euler-Maruyama). 0 = deterministic.
    dt_sde : float, optional
        Time step for Euler-Maruyama when sigma > 0.

    Returns
    -------
    If single controller: results_dict (as before)
    If list of controllers: dict with keys as controller names, values as results_dict
    '''

    if dist is None and X0_distance is not None:
        dist = X0_distance

    sample_kw = {"dist": dist}
    if K is not None:
        sample_kw["K"] = K

    if isinstance(controller, list):
        if X0_pool is None:
            if n_MC is None:
                raise ValueError("n_MC must be set by the experiment (e.g. n_MC=100)")
            np.random.seed(random_seed)
            X0_pool = sampling.sample_conditions(config, n_MC, **sample_kw)
        
        # Evaluate each controller with the same X0_pool
        all_results = {}
        for name, ctrl in controller:
            if verbose:
                print(f"\nEvaluating {name}...")
            results = monte_carlo(
                OCP, config, ctrl,
                X0_distance=X0_distance, random_seed=None, X0_pool=X0_pool,
                n_MC=n_MC, dist=dist, K=K,
                solve_open_loop=solve_open_loop, verbose=verbose, suppress_warnings=suppress_warnings,
                sigma=sigma, dt_sde=dt_sde
            )
            all_results[name] = results
        
        return all_results

    if X0_pool is None:
        if n_MC is None:
            raise ValueError("n_MC must be set by the experiment (e.g. n_MC=100)")
        np.random.seed(random_seed)
        X0_pool = sampling.sample_conditions(config, n_MC, **sample_kw)
    else:
        # If X0_pool is provided, use its actual size
        n_MC = X0_pool.shape[1]  # X0_pool is (n_states, n_samples)

    ocp_converged = np.zeros(n_MC, dtype=bool)
    init_dists = np.empty(n_MC)
    final_dists = np.empty(n_MC)
    # ... rest of the code

    NN_final_times = np.full_like(init_dists, np.inf)
    NN_costs = np.full_like(init_dists, np.inf)

    # Default final time value is -1
    # Not possible under normal circumstances, helps with catching errors later
    opt_final_times = np.full_like(NN_final_times, -1.)
    opt_costs = np.full_like(NN_costs, -1.)

    def converged(X, U):
        return OCP.convergence(X, U, config.conv_tol, config.fp_tol)

    events = OCP.make_integration_events()

    # ------------------------------------------------------------------------ #

    for i in tqdm(range(n_MC)):
        X0 = X0_pool[:,i].flatten()  # Ensure 1D array
        
        if not _is_finite(X0):
            init_dists[i] = np.nan
            continue
        
        init_dists[i] = OCP.norm(X0)
        
        simulation_failed = False
        try:
            t, X, ode_converged = sim_to_converge(
                OCP.dynamics, OCP.closed_loop_jacobian, controller, X0, config,
                events=events, sigma=sigma, dt_sde=dt_sde
            )
            if not _is_finite(X):
                simulation_failed = True
        except (ValueError, RuntimeError):
            simulation_failed = True

        if simulation_failed:
            continue
        
        try:
            V, dVdX, U = controller.bvp_guess(X)
        except Exception:
            V, dVdX, U = None, None, None

        if U is None:
            continue
        U = np.asarray(U)
        if U.size == 0 or not _is_finite(U):
            continue

        # Ensure U is (n_controls, n_timepoints)
        if U.ndim == 1:
            if U.shape[0] == OCP.n_controls:
                U = U.reshape(-1, 1)
            else:
                continue
        elif U.ndim == 2 and U.shape[0] != OCP.n_controls:
            continue
        elif U.ndim > 2:
            continue
        
        k, converged_flag = clip_trajectory(t, X, U, converged)

        if ode_converged and converged_flag and X.shape[1] > 0:
            final_dists_arr = OCP.norm(X[:, -1])
            final_dists[i] = float(final_dists_arr.flat[0] if final_dists_arr.size > 0 else 0.0)
            NN_final_times[i] = t[k]
            try:
                NN_costs[i] = OCP.compute_cost(t, X, U).flatten()[-1]
                if not np.isfinite(NN_costs[i]):
                    NN_final_times[i] = np.inf
                    NN_costs[i] = np.inf
            except Exception:
                NN_final_times[i] = np.inf
                NN_costs[i] = np.inf
        else:
            final_dists[i] = np.inf

        # -------------------------------------------------------------------- #

        if solve_open_loop and V is not None:
            try:
                ocp_sol, cont_ocp_sol, ocp_converged[i] = solve_ocp(
                    OCP, config,
                    t_guess=t, X_guess=X, U_guess=U, dVdX_guess=dVdX, V_guess=V,
                    solve_to_converge=True,
                    verbose=verbose, suppress_warnings=suppress_warnings
                )


                t = np.concatenate((ocp_sol['t'].flatten(), t.flatten()))
                t = np.unique(t)
                ocp_sol = cont_ocp_sol(t)

                k, _ = clip_trajectory(
                    t, ocp_sol['X'], ocp_sol['U'], converged
                )

                opt_costs[i] = ocp_sol['V'].flatten()[0]
                opt_final_times[i] = t[k]

                            # ... rest of open loop code
            except Exception:
                # OCP solve failed - mark as not converged
                ocp_converged[i] = False



    results_dict = {
        'seed': random_seed,
        #'X0_pool': X0_pool,
        'init_dists': init_dists,
        'final_dists': final_dists,
        'NN_final_times': NN_final_times,
        'NN_costs': NN_costs,
        'opt_final_times': opt_final_times,
        'opt_costs': opt_costs,
        'ocp_converged': ocp_converged
    }
    if random_seed is None:
        results_dict['seed'] = -1

    if solve_open_loop:
        print('\n%d/%d OCPs converged.\n' % (ocp_converged.sum(), n_MC))

    return results_dict


def monte_carlo_nu_mismatch(
    OCP_baseline,
    config_baseline,
    controllers,
    nu_eval,
    n_MC=None,
    dist=None,
    K=None,
    random_seed=None,
    verbose=0,
    suppress_warnings=True,
    **monte_carlo_kw
):
    """
    Run Monte Carlo with a model mismatch: dynamics use nu=nu_eval instead of
    the baseline nu. Controllers are unchanged (trained on baseline). Useful
    for robustness evaluation (e.g. nu_eval < baseline nu is typically harder).

    Parameters
    ----------
    OCP_baseline : OCP instance (unused; eval OCP is built from config)
    config_baseline : config with .system, .n_states, .n_controls, .seed, etc.
    controllers : list of (name, controller) or single controller
    nu_eval : float
        Diffusion coefficient used in the evaluation OCP.
    n_MC, dist, K, random_seed, verbose, suppress_warnings : passed to monte_carlo
    **monte_carlo_kw : additional kwargs for monte_carlo

    Returns
    -------
    Same as monte_carlo (dict of results per controller, or single results_dict)
    """
    from problems import create_config
    overrides = {
        "system": config_baseline.system,
        "nu": nu_eval,
        "n_states": getattr(config_baseline, "n_states", None),
        "n_controls": getattr(config_baseline, "n_controls", None),
        "seed": getattr(config_baseline, "seed", None),
        "t1_initial": getattr(config_baseline, "t1_initial", None),
        "t1_max": getattr(config_baseline, "t1_max", None),
        "fp_tol": getattr(config_baseline, "fp_tol", None),
        "conv_tol": getattr(config_baseline, "conv_tol", None),
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    config_eval = create_config(**overrides)
    return monte_carlo(
        config_eval.ocp,
        config_eval,
        controllers,
        n_MC=n_MC,
        dist=dist,
        K=K,
        random_seed=random_seed,
        verbose=verbose,
        suppress_warnings=suppress_warnings,
        **monte_carlo_kw
    )


from copy import deepcopy



class OpenLoopSolver:
    def __init__(self, OCP, **kwargs):
        self.OCP = OCP
        self.sol = {}

    def solve(self, t=None, X=None, U=None, dVdX=None, V=None, verbose=0):
        raise NotImplementedError

    def continuous_sol(self, t):
        raise NotImplementedError

    def check_converged(self, conv_tol, fp_tol):
        raise NotImplementedError

    def extend_horizon(self):
        raise NotImplementedError

class IndirectSolver(OpenLoopSolver):
    def __init__(
            self, OCP, tol=1e-05, t1_scale=3/2, t1_max=np.inf, max_nodes=1000
        ):
        self.tol = tol
        self.t1_scale = t1_scale
        self.t1_max = t1_max
        self.max_nodes = max_nodes

        self.bvp_sol = None
        self.bc = None

        super().__init__(OCP)

    def solve(self, t=None, X=None, U=None, dVdX=None, V=None, verbose=0):
        X = X.reshape(self.OCP.n_states, -1)
        dVdX = dVdX.reshape(self.OCP.n_states, -1)
        V = V.reshape(1, -1)

        X_aug = np.vstack((X, dVdX, V))

        if self.bc is None:
            self.bc = self.OCP.make_bc(X[:,0])

        self.bvp_sol = solve_bvp(
            self.OCP.bvp_dynamics, self.bc, t, X_aug,
            tol=self.tol, max_nodes=self.max_nodes, verbose=verbose
        )

        self.sol['t'] = self.bvp_sol.x
        self.sol['X'] = self.bvp_sol.y[:self.OCP.n_states]
        self.sol['dVdX'] = self.bvp_sol.y[self.OCP.n_states:-1]
        self.sol['V'] = self.bvp_sol.y[-1:]
        self.sol['U'] = self.OCP.U_star(self.sol['X'], self.sol['dVdX'])

    def continuous_sol(self, t):
        X_aug = self.bvp_sol.sol(t)

        X = X_aug[:self.OCP.n_states]
        dVdX = X_aug[self.OCP.n_states:-1]
        V = X_aug[-1:]

        U = self.OCP.U_star(X, dVdX)

        return {'X': X, 'U': U, 'dVdX': dVdX, 'V': V}

    def check_converged(self, conv_tol, fp_tol):
        """True if running cost < conv_tol and dynamics norm < fp_tol at final time."""
        if self.bvp_sol is None or not self.bvp_sol.success:
            return False

        L = self.OCP.running_cost(self.sol['X'][:,-1], self.sol['U'][:,-1])
        F = self.OCP.bvp_dynamics(self.sol['t'][-1:], self.bvp_sol.y[:,-1:])
        F = np.linalg.norm(F[:self.OCP.n_states])

        return L <= conv_tol and F <= fp_tol

    def extend_horizon(self):
        if self.bvp_sol is None:
            return False
        # Cannot extend horizon if exceeded number of mesh nodes or maximum time
        if self.bvp_sol.status == 1 or self.sol['t'][-1] >= self.t1_max:
            return False

        self.sol['t'][-1] = np.minimum(
            self.t1_max, self.sol['t'][-1]*self.t1_scale
        )

        return True


# ---------------------------------------------------------------------------- #

def solve_ocp(
        OCP, config,
        t_guess=None, X_guess=None, U_guess=None, dVdX_guess=None, V_guess=None,
        solve_to_converge=False, verbose=0, suppress_warnings=True
    ):
    if config.ocp_solver == 'indirect':
        solver = IndirectSolver(
            OCP,
            tol=config.indirect_tol,
            t1_scale=config.t1_scale,
            t1_max=config.t1_max,
            max_nodes=config.indirect_max_nodes
        )
    elif config.ocp_solver == 'direct':
        solver = DirectSolver(
            OCP,
            tol=config.direct_tol,
            tol_scale=config.direct_tol_scale,
            n_init_nodes=config.direct_n_init_nodes,
            n_add_nodes=config.direct_n_add_nodes,
            max_nodes=config.direct_max_nodes,
            max_iter=config.direct_max_slsqp_iter
        )
    elif config.ocp_solver == 'direct_to_indirect':
        config_copy = deepcopy(config)
        config_copy.ocp_solver = 'direct'

        direct_start, _, _ = solve_ocp(
            OCP,
            config_copy,
            t_guess=t_guess,
            X_guess=X_guess,
            U_guess=U_guess,
            dVdX_guess=dVdX_guess,
            V_guess=V_guess,
            solve_to_converge=True,
            verbose=verbose,
            suppress_warnings=suppress_warnings
        )

        config_copy.ocp_solver = 'indirect'

        idx = direct_start['t'] <= t_guess[-1]
        dVdX_guess = direct_start['dVdX'][:,idx]
        dVdX_guess[:,-1] = 0.

        return solve_ocp(
            OCP,
            config_copy,
            t_guess=direct_start['t'][idx],
            X_guess=direct_start['X'][:,idx],
            U_guess=direct_start['U'][:,idx],
            dVdX_guess=dVdX_guess,
            V_guess=direct_start['V'][:,idx],
            solve_to_converge=solve_to_converge,
            verbose=verbose,
            suppress_warnings=suppress_warnings
        )
    else:
        raise ValueError(
            'config.ocp_solver must be one of "direct", "indirect", or "direct_to_indirect"'
        )

    def _converged():
        return solver.check_converged(conv_tol=config.conv_tol, fp_tol=config.fp_tol)

    with warnings.catch_warnings():
        if suppress_warnings:
            np.seterr(over='warn', divide='warn', invalid='warn')
            warnings.filterwarnings("ignore", category=RuntimeWarning)

        solver.solve(
            t=t_guess, X=X_guess, U=U_guess, dVdX=dVdX_guess, V=V_guess,
            verbose=verbose
        )
        converged = _converged()

        # Solves the OCP over an extended time interval until convergece
        # conditions are satisfied
        if solve_to_converge and not converged:
            if suppress_warnings:
                # If we don't want to see warnings, treat these as errors so the
                # try context will catch them and mark the trajectory as not
                # converged instead of printing out the warning.
                warnings.filterwarnings("error", category=RuntimeWarning)

            try:
                while not converged and solver.extend_horizon():
                    solver.solve(**solver.sol, verbose=verbose)
                    converged = _converged()
            except RuntimeWarning:
                pass

        return solver.sol, solver.continuous_sol, converged

# ---------------------------------------------------------------------------- #

def clip_trajectory(t, X, U, criteria):
    '''
    Go backwards in time and check to see when a function (typically the
    running cost or vector field norm) is sufficiently small.

    Parameters
    ----------
    criteria : callable

    Returns
    -------
    converged_idx : int
        Integer such that X[:,converged_idx], U[:,converged_idx] is the first
        state-control pair for which criteria(X, U) < tol.
    converged : bool
        True if some pair X, U satisfied criteria(X, U) < tol, False if no
        such pair was found.
    '''
    converged_idx = criteria(X, U).flatten()

    converged = converged_idx.any()

    if converged:
        converged_idx = np.min(np.argwhere(converged_idx))
    else:
        converged_idx = t.shape[0] - 1

    return converged_idx, converged


def process_robustness_results(robustness_results, controller_names, perturbation_variants=None):
    """
    Process robustness evaluation results into a summary DataFrame.

    Parameters
    ----------
    robustness_results : dict
        Dictionary keyed by perturbation names, where each value is a dict
        keyed by controller names, containing monte_carlo results.
    controller_names : list
        List of controller names to include in the summary.
    perturbation_variants : list, optional
        List of (pert_type, strength, display_name), e.g.
        [("advection", 0.10, "Advection (c=0.10)"), ("advection", 0.15, "Advection (c=0.15)")].
        If given, columns are ordered by this list and returned with MultiIndex
        (Stability/Cost, c) for a lean table. If None, columns are flat per pert_name.

    Returns
    -------
    pd.DataFrame
        Rows = controllers. Columns: if perturbation_variants is given,
        MultiIndex [("Stability $S$", c1), ..., ("Cost $J$", c1), ...];
        else flat "PertName Stab." / "PertName Cost".
    """
    import pandas as pd

    if perturbation_variants is not None:
        c_values = [strength for (_, strength, _) in perturbation_variants]
        summary_rows = []
        for ctrl_name in controller_names:
            row = {"Controller": ctrl_name}
            for (_, c, pert_name) in perturbation_variants:
                results = robustness_results.get(pert_name)
                if results is None or ctrl_name not in results:
                    row[("Stability $S$", c)] = "N/A"
                    row[("Cost $J$", c)] = "N/A"
                    continue
                ctrl_results = results[ctrl_name]
                NN_final_times = np.asarray(ctrl_results["NN_final_times"])
                NN_costs = np.asarray(ctrl_results["NN_costs"])
                converged_mask = np.isfinite(NN_final_times)
                n_converged = np.sum(converged_mask)
                n_total = len(NN_final_times)
                stability_rate = n_converged / n_total if n_total > 0 else 0.0
                mean_cost = np.mean(NN_costs[converged_mask]) if n_converged > 0 else np.nan
                row[("Stability $S$", c)] = f"{stability_rate * 100:.0f}%"
                row[("Cost $J$", c)] = f"{mean_cost:.2f}" if mean_cost < 1e6 and not np.isnan(mean_cost) else "---"
            summary_rows.append(row)
        df = pd.DataFrame(summary_rows)
        df = df.set_index("Controller")
        df.columns = pd.MultiIndex.from_tuples(
            [("Stability $S$", c) for c in c_values] + [("Cost $J$", c) for c in c_values]
        )
        return df

    summary_rows = []
    for ctrl_name in controller_names:
        row = {"Controller": ctrl_name}
        for pert_name, results in robustness_results.items():
            if ctrl_name not in results:
                row[f"{pert_name} Stab."] = "N/A"
                row[f"{pert_name} Cost"] = "N/A"
                continue
            ctrl_results = results[ctrl_name]
            NN_final_times = np.asarray(ctrl_results["NN_final_times"])
            NN_costs = np.asarray(ctrl_results["NN_costs"])
            converged_mask = np.isfinite(NN_final_times)
            n_converged = np.sum(converged_mask)
            n_total = len(NN_final_times)
            stability_rate = n_converged / n_total if n_total > 0 else 0.0
            mean_cost = np.mean(NN_costs[converged_mask]) if n_converged > 0 else np.nan
            row[f"{pert_name} Stab."] = f"{stability_rate * 100:.0f}%"
            row[f"{pert_name} Cost"] = f"{mean_cost:.2f}" if mean_cost < 1e6 and not np.isnan(mean_cost) else "---"
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)

