import inspect
import warnings
import numpy as np

from scipy.integrate._ivp.ivp import METHODS, MESSAGES
from scipy.integrate._ivp.ivp import OdeResult, OdeSolution, OdeSolver
from scipy.integrate import solve_bvp
import sampling


def solve_ivp(
        fun, t_span, y0, method='RK45', t_eval=None, dense_output=False,
        events=None, vectorized=False, args=None, **options
    ):
    """Solve an initial value problem for a system of ODEs.

    Modification of scipy.integrate.solve_ivp check events only after each
    solver step without root-finding. This makes event times less accurate but
    computations faster and more robust.

    This function numerically integrates a system of ordinary differential
    equations given an initial value:
        dy / dt = f(t, y)
        y(t0) = y0
    Here t is a 1-D independent variable (time), y(t) is an N-D vector-valued
    function (state), and an N-D vector-valued function f(t, y) determines the
    differential equations. The goal is to find y(t) approximately satisfying
    the differential equations, given an initial value y(t0)=y0. Some of the
    solvers support integration in the complex domain, but note that for stiff
    ODE solvers, the right-hand side must be complex-differentiable (satisfy
    Cauchy-Riemann equations [11]_). To solve a problem in the complex domain,
    pass y0 with a complex data type. Another option always available is to
    rewrite your problem for real and imaginary parts separately.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here `t` is a scalar, and there are two options for the ndarray `y`: It
        can either have shape (n,); then `fun` must return array_like with shape
        (n,). Alternatively, it can have shape (n, k); then `fun` must return an
        array_like with shape (n, k), i.e., each column corresponds to a single
        column in `y`. The choice between the two options is determined by
        `vectorized` argument (see below). The vectorized implementation allows
        a faster approximation of the Jacobian by finite differences (required
        for stiff solvers).
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf.
    y0 : array_like, shape (n,)
        Initial state. For problems in the complex domain, pass `y0` with a
        complex data type (even if the initial value is purely real).
    method : string or `OdeSolver`, optional
        Integration method to use:
            * 'RK45' (default): Explicit Runge-Kutta method of order 5(4) [1]_.
              The error is controlled assuming accuracy of the fourth-order
              method, but steps are taken using the fifth-order accurate formula
              (local extrapolation is done). A quartic interpolation polynomial
              is used for the dense output [2]_. Can be applied in the complex
              domain.
            * 'RK23': Explicit Runge-Kutta method of order 3(2) [3]_. The error
              is controlled assuming accuracy of the second-order method, but
              steps are taken using the third-order accurate formula (local
              extrapolation is done). A cubic Hermite polynomial is used for the
              dense output. Can be applied in the complex domain.
            * 'DOP853': Explicit Runge-Kutta method of order 8 [13]_.
              Python implementation of the "DOP853" algorithm originally written
              in Fortran [14]_. A 7-th order interpolation polynomial accurate
              to 7-th order is used for the dense output. Can be applied in the
              complex domain.
            * 'Radau': Implicit Runge-Kutta method of the Radau IIA family of
              order 5 [4]_. The error is controlled with a third-order accurate
              embedded formula. A cubic polynomial which satisfies the
              collocation conditions is used for the dense output.
            * 'BDF': Implicit multi-step variable-order (1 to 5) method based
              on a backward differentiation formula for the derivative
              approximation [5]_. The implementation follows the one described
              in [6]_. A quasi-constant step scheme is used and accuracy is
              enhanced using the NDF modification. Can be applied in the complex
              domain.
            * 'LSODA': Adams/BDF method with automatic stiffness detection and
              switching [7]_, [8]_. This is a wrapper of the Fortran solver from
              ODEPACK.
        You can also pass an arbitrary class derived from `OdeSolver` which
        implements the solver.
    t_eval : array_like, optional
        Times at which to store the computed solution, must be sorted and lie
        within `t_span`. If None (default), use points selected by the solver.
    dense_output : bool, optional
        Whether to compute a continuous solution. Default is False.
    events : callable, or list of callables, optional
        Events to track. If None (default), no events will be tracked. Each
        function must have the signature ``event(t, y)`` and return a float. The
        solver looks for a sign change over each time step, so if multiple zero
        crossings occur within one step, events may be missed. Note that unlike
        scipy.integrate.solve_ivp, the event time is not refined exactly with
        any root-finding. Each `event` function might also have the following
        attribute:
            terminal: bool, optional
                Whether to terminate integration if this event occurs.
                Implicitly False if not assigned.
            direction: float, optional
                Direction of a zero crossing. If `direction` is positive,
                `event` will only trigger when going from negative to positive,
                and vice versa if `direction` is negative. If 0, then either
                direction will trigger event. Implicitly 0 if not assigned.
        You can assign attributes like ``event.terminal = True`` to any
        function in Python.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.
    args : tuple, optional
        Additional arguments to pass to the user-defined functions.  If given,
        the additional arguments are passed to all user-defined functions.
        So if, for example, `fun` has the signature ``fun(t, y, a, b, c)``,
        then `jac` (if given) and any event functions must have the same
        signature, and `args` must be a tuple of length 3.
    options
        Options passed to a chosen solver. All options available for already
        implemented solvers are listed below.
    first_step : float or None, optional
        Initial step size. Default is `None` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float or array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be lower than the lowest value that can
        be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always lower
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    jac : array_like, sparse_matrix, callable or None, optional
        Jacobian matrix of the right-hand side of the system with respect
        to y, required by the 'Radau', 'BDF' and 'LSODA' method. The
        Jacobian matrix has shape (n, n) and its element (i, j) is equal to
        ``d f_i / d y_j``.  There are three ways to define the Jacobian:
            * If array_like or sparse_matrix, the Jacobian is assumed to
              be constant. Not supported by 'LSODA'.
            * If callable, the Jacobian is assumed to depend on both
              t and y; it will be called as ``jac(t, y)``, as necessary.
              For 'Radau' and 'BDF' methods, the return value might be a
              sparse matrix.
            * If None (default), the Jacobian will be approximated by
              finite differences.
        It is generally recommended to provide the Jacobian rather than
        relying on a finite-difference approximation.
    jac_sparsity : array_like, sparse matrix or None, optional
        Defines a sparsity structure of the Jacobian matrix for a finite-
        difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few
        non-zero elements in *each* row, providing the sparsity structure
        will greatly speed up the computations [10]_. A zero entry means that
        a corresponding element in the Jacobian is always zero. If None
        (default), the Jacobian is assumed to be dense.
        Not supported by 'LSODA', see `lband` and `uband` instead.
    lband, uband : int or None, optional
        Parameters defining the bandwidth of the Jacobian for the 'LSODA'
        method, i.e., ``jac[i, j] != 0 only for i - lband <= j <= i + uband``.
        Default is None. Setting these requires your jac routine to return the
        Jacobian in the packed format: the returned array must have ``n``
        columns and ``uband + lband + 1`` rows in which Jacobian diagonals are
        written. Specifically ``jac_packed[uband + i - j , j] = jac[i, j]``.
        The same format is used in `scipy.linalg.solve_banded` (check for an
        illustration).  These parameters can be also used with ``jac=None`` to
        reduce the number of Jacobian elements estimated by finite differences.
    min_step : float, optional
        The minimum allowed step size for 'LSODA' method.
        By default `min_step` is zero.

    Returns
    -------
    Bunch object with the following fields defined:
    t : ndarray, shape (n_points,)
        Time points.
    y : ndarray, shape (n, n_points)
        Values of the solution at `t`.
    sol : `OdeSolution` or None
        Found solution as `OdeSolution` instance; None if `dense_output` was
        set to False.
    t_events : list of ndarray or None
        Contains for each event type a list of arrays at which an event of
        that type event was detected. None if `events` was None.
    y_events : list of ndarray or None
        For each value of `t_events`, the corresponding value of the solution.
        None if `events` was None.
    nfev : int
        Number of evaluations of the right-hand side.
    njev : int
        Number of evaluations of the Jacobian.
    nlu : int
        Number of LU decompositions.
    status : int
        Reason for algorithm termination:
            * -1: Integration step failed.
            *  0: The solver successfully reached the end of `tspan`.
            *  1: A termination event occurred.
    message : string
        Human-readable description of the termination reason.
    success : bool
        True if the solver reached the interval end or a termination event
        occurred (``status >= 0``).
    """
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
        solver='LSODA', atol=1e-06, rtol=1e-03
    ):
    '''
    Simulate the closed-loop system for a fixed time interval.

    Parameters
    ----------
        OCP: instance of TemplateOCP defining dynamics, Jacobian, etc.
        tspan: integration time, list of two floats
        X0: initial condition, (n,) numpy array
        controller: instance of a trained QRnet
        solver: ODE solver to use, str
        atol: absolute integration tolerance, float
        rtol: relative integration tolerance, float
        t_eval: optional (Nt,) numpy array of time instances to evaluate solution at

    Returns
    -------
        t: time vector, (Nt,) numpy array
        X: state time series, (n,Nt) numpy array
    '''
    def dynamics_wrapper(t, X):
        U = controller.eval_U(X)
        return dynamics(X, U)

    #def jac_wrapper(t, X):
        #eturn jacobian(X, controller)

    ode_sol = solve_ivp(
        dynamics_wrapper, tspan, X0, t_eval=t_eval, jac=None,
        events=events, vectorized=True, method=solver, rtol=rtol, atol=atol
    )

    return ode_sol.t, ode_sol.y, ode_sol.status

def sim_to_converge(
        dynamics, jacobian, controller, X0, config, events=None
    ):
    '''
    Simulate the closed-loop system until reach t_max or the dX/dt = 0.

    Parameters
    ----------
        OCP: instance of a setupProblem class defining dynamics, Jacobian, etc.
        config: a configuration dict defined in problem_def.py
        X0: initial condition, (n,) numpy array
        controller: instance of a trained QRnet

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
        # Validate current state before simulation
        X_current = X[:,-1].flatten()
        if not np.isfinite(X_current).all():
            # State became non-finite, simulation failed
            break
        
        t1 = np.maximum(config.t1_initial, t[-1] * config.t1_scale)
        
        try:
            # Simulate the closed-loop system
            t_new, X_new, status = sim_closed_loop(
                dynamics,
                jacobian,
                controller,
                [t[-1], t1],
                X_current,
                events=events,
                solver='LSODA',
                atol=1e-06,
                rtol=1e-03
            )
        except (ValueError, RuntimeError) as e:
            # Simulation failed (likely non-finite state)
            break

        # Validate the new state
        if not np.isfinite(X_new).all():
            # Simulation produced non-finite values, stop
            break

        t = np.concatenate((t, t_new[1:]))
        X = np.hstack((X, X_new[:,1:]))

        if status == 1:
            break

        # Validate state before computing convergence
        if not np.isfinite(X[:,-1]).all():
            break
            
        try:
            U = controller.eval_U(X[:,-1])
            if not np.isfinite(U).all():
                break
            converged = np.linalg.norm(dynamics(X[:,-1], U)) < config.fp_tol
        except (ValueError, RuntimeError):
            # Failed to compute control or dynamics
            break

    return t, X, converged

import numpy as np
from tqdm import tqdm



def monte_carlo(
        OCP, config, controller,
        X0_distance=None, random_seed=None, X0_pool=None,
        n_MC=100, dist=None,
        solve_open_loop=False, verbose=0, suppress_warnings=True
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
    n_MC : int, default=25
        Number of Monte Carlo samples to evaluate
    dist : float, optional
        Distance/norm to use for random initial conditions
    solve_open_loop : bool, default=False
        Set to True to solve the open loop OCP for each initial condition.
    verbose : int, default=0
        Verbosity level
    suppress_warnings : bool, default=True
        If True, treat numpy warnings as OCP failures.

    Returns
    -------
    If single controller: results_dict (as before)
    If list of controllers: dict with keys as controller names, values as results_dict
    '''

    # Use X0_distance if dist is not provided (backward compatibility)
    if dist is None and X0_distance is not None:
        dist = X0_distance

    # Check if controller is a list of (name, controller) tuples
    if isinstance(controller, list):
        # Generate X0_pool once for all controllers
        if X0_pool is None:
            np.random.seed(random_seed)
            X0_pool = sampling.sample_conditions(config, n_MC, dist=dist)
        
        # Evaluate each controller with the same X0_pool
        all_results = {}
        for name, ctrl in controller:
            if verbose:
                print(f"\nEvaluating {name}...")
            results = monte_carlo(
                OCP, config, ctrl,
                X0_distance=X0_distance, random_seed=None, X0_pool=X0_pool,
                n_MC=n_MC, dist=dist,
                solve_open_loop=solve_open_loop, verbose=verbose, suppress_warnings=suppress_warnings
            )
            all_results[name] = results
        
        return all_results

    # Original single controller code continues below...
    # Set n_MC from X0_pool if provided, otherwise use parameter/default
    if X0_pool is None:
        np.random.seed(random_seed)
        X0_pool = sampling.sample_conditions(config, n_MC, dist=dist)
        bad = ~np.isfinite(X0_pool)
        bad_cols = np.where(bad.any(axis=0))[0]
        bad_cols[:10], len(bad_cols)
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

    def closed_converged(X, U):
        # Check both: dynamics are small AND state is near zero
        dynamics_small = np.linalg.norm(OCP.dynamics(X, U), axis=0) < config.fp_tol
        state_near_zero = OCP.norm(X) < 0.01  # Threshold for "near zero" - adjust as needed
        return dynamics_small & state_near_zero
    def open_converged(X, U):
        return OCP.running_cost(X, U) < config.fp_tol

    events = OCP.make_integration_events()

    # ------------------------------------------------------------------------ #

    for i in tqdm(range(n_MC)):
        X0 = X0_pool[:,i].flatten()  # Ensure 1D array
        
        # Validate X0 before simulation
        if not np.isfinite(X0).all():
            # Invalid X0 - mark as failed but don't skip
            init_dists[i] = np.nan
            # final_dists, NN_final_times, NN_costs already set to inf/nan
            continue
        
        init_dists[i] = OCP.norm(X0)
        
        # Integrates the closed-loop system
        simulation_failed = False
        try:
            t, X, ode_converged = sim_to_converge(
                OCP.dynamics, OCP.closed_loop_jacobian, controller, X0, config,
                events=events
            )
            
            # Check if simulation produced valid results
            if not np.isfinite(X).all():
                # Simulation produced non-finite values - difficult IC
                simulation_failed = True
                
        except (ValueError, RuntimeError) as e:
            # Simulation failed - difficult IC, not an error
            simulation_failed = True
        
        if simulation_failed:
            # Mark as not converged (values already inf/nan)
            # This is just a difficult initial condition
            continue
        
        # If we get here, simulation succeeded
        try:
            V, dVdX, U = controller.bvp_guess(X)
        except Exception as e:
            # bvp_guess failed - still count as simulation succeeded but can't use for OCP
            V, dVdX, U = None, None, None
        
        # Check if controls are valid before proceeding
        # Invalid controls indicate the controller failed, so don't count as converged
        if U is None:
            # Invalid controls - mark as not converged
            # final_dists, NN_final_times, NN_costs already set to inf/nan
            continue
        
        # Ensure U is a numpy array and check for invalid values
        U = np.asarray(U)
        if U.size == 0 or not np.isfinite(U).all():
            # Invalid controls - mark as not converged
            continue
        
        # Check if U has the right shape (should be (n_controls, n_timepoints) or (n_controls,))
        # bvp_guess should return U with shape matching X: (n_controls, n_timepoints)
        if U.ndim == 1:
            # Reshape to (n_controls, 1) for consistency if it's a single control vector
            if U.shape[0] == OCP.n_controls:
                U = U.reshape(-1, 1)
            else:
                # Wrong number of controls - mark as not converged
                continue
        elif U.ndim == 2:
            if U.shape[0] != OCP.n_controls:
                # Wrong number of controls - mark as not converged
                continue
            # U.shape[1] should match X.shape[1] (number of time points), but we'll let
            # clip_trajectory handle that - if shapes don't match, it will fail gracefully
        else:
            # Invalid dimensionality - mark as not converged
            continue
        
        k, converged_flag = clip_trajectory(t, X, U, closed_converged)
        

        
        # Only mark as converged if both ode_converged AND converged_flag are True
        # This ensures the trajectory actually reached a valid equilibrium with valid controls
        if ode_converged and converged_flag:
            # Use the final state of the trajectory (X[:,-1]) which should be closest to equilibrium
            # The trajectory continues after the first convergence point, so the final state is more accurate
            if X.shape[1] > 0:
                X_final = X[:,-1]  # Final state of the trajectory
                final_dists_arr = OCP.norm(X_final)
                # Extract scalar from array
                final_dists[i] = float(final_dists_arr.flat[0] if final_dists_arr.size > 0 else 0.0)
                

            else:
                final_dists[i] = np.inf
            
            NN_final_times[i] = t[k]
            try:
                NN_costs[i] = OCP.compute_cost(t, X, U).flatten()[-1]
                # Double-check that cost is valid
                if not np.isfinite(NN_costs[i]):
                    NN_final_times[i] = np.inf
                    NN_costs[i] = np.inf
            except Exception:
                # Cost computation failed - mark as not converged
                NN_final_times[i] = np.inf
                NN_costs[i] = np.inf
        else:
            # Not converged - ensure final_dists is set to inf (not uninitialized garbage)
            final_dists[i] = np.inf
            # DEBUG
            if i < 3:
                print(f"Trajectory {i} NOT converged - final_dists[{i}] = inf")

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
                    t, ocp_sol['X'], ocp_sol['U'], open_converged
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


from copy import deepcopy



class OpenLoopSolver:
    def __init__(self, OCP, **kwargs):
        self.OCP = OCP
        self.sol = {}

    def solve(self, t=None, X=None, U=None, dVdX=None, V=None, verbose=0):
        raise NotImplementedError

    def continuous_sol(self, t):
        raise NotImplementedError

    def check_converged(self, tol):
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

    def check_converged(self, tol):
        '''
        Check if the running cost L and vector field F at final time are smaller
        than a given tolerance, to see if the BVP is converged.

        Parameters
        ----------

        Returns
        -------
        '''
        if self.bvp_sol is None or not self.bvp_sol.success:
            return False

        L = self.OCP.running_cost(self.sol['X'][:,-1], self.sol['U'][:,-1])
        F = self.OCP.bvp_dynamics(self.sol['t'][-1:], self.bvp_sol.y[:,-1:])
        F = np.linalg.norm(F[:self.OCP.n_states])

        return L <= tol and F <= tol

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
        if getattr(config, 'system', None) == 'eikonal':
            # Stationary HJB: success + small residual; skip L<=tol (running cost)
            if solver.bvp_sol is None or not solver.bvp_sol.success:
                return False
            F = OCP.bvp_dynamics(solver.sol['t'][-1:], solver.bvp_sol.y[:, -1:])
            return np.linalg.norm(F[:OCP.n_states]) <= config.fp_tol
        return solver.check_converged(config.fp_tol)

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


def process_robustness_results(robustness_results, controller_names):
    """
    Process robustness evaluation results into a summary DataFrame.
    
    Parameters
    ----------
    robustness_results : dict
        Dictionary keyed by perturbation names, where each value is a dict
        keyed by controller names, containing monte_carlo results.
    controller_names : list
        List of controller names to include in the summary.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with rows for each controller and columns for stability
        and cost for each perturbation variant.
    """
    import pandas as pd
    
    summary_rows = []
    
    for ctrl_name in controller_names:
        row = {"Controller": ctrl_name}
        for pert_name, results in robustness_results.items():
            # results is a dict keyed by controller names (from monte_carlo)
            if ctrl_name not in results:
                # Controller not evaluated for this perturbation
                row[f"{pert_name} Stab."] = "N/A"
                row[f"{pert_name} Cost"] = "N/A"
                continue
                
            ctrl_results = results[ctrl_name]
            
            # Compute stats from results dict (similar to _compute_monte_carlo_stats)
            NN_final_times = np.asarray(ctrl_results['NN_final_times'])
            NN_costs = np.asarray(ctrl_results['NN_costs'])
            
            converged_mask = np.isfinite(NN_final_times)
            n_converged = np.sum(converged_mask)
            n_total = len(NN_final_times)
            stability_rate = n_converged / n_total if n_total > 0 else 0.0
            mean_cost = np.mean(NN_costs[converged_mask]) if n_converged > 0 else np.nan
            
            stability = stability_rate * 100
            cost = mean_cost
            row[f"{pert_name} Stab."] = f"{stability:.0f}%"
            row[f"{pert_name} Cost"] = f"{cost:.2f}" if cost < 1e6 and not np.isnan(cost) else "—"
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def process_robustness_results(robustness_results, controller_names):
    """
    Process robustness evaluation results into a summary DataFrame.
    
    Parameters
    ----------
    robustness_results : dict
        Dictionary keyed by perturbation names, where each value is a dict
        keyed by controller names, containing monte_carlo results.
    controller_names : list
        List of controller names to include in the summary.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with rows for each controller and columns for stability
        and cost for each perturbation variant.
    """
    import pandas as pd
    
    summary_rows = []
    
    for ctrl_name in controller_names:
        row = {"Controller": ctrl_name}
        for pert_name, results in robustness_results.items():
            # results is a dict keyed by controller names (from monte_carlo)
            if ctrl_name not in results:
                # Controller not evaluated for this perturbation
                row[f"{pert_name} Stab."] = "N/A"
                row[f"{pert_name} Cost"] = "N/A"
                continue
                
            ctrl_results = results[ctrl_name]
            
            # Compute stats from results dict (similar to _compute_monte_carlo_stats)
            NN_final_times = np.asarray(ctrl_results['NN_final_times'])
            NN_costs = np.asarray(ctrl_results['NN_costs'])
            
            converged_mask = np.isfinite(NN_final_times)
            n_converged = np.sum(converged_mask)
            n_total = len(NN_final_times)
            stability_rate = n_converged / n_total if n_total > 0 else 0.0
            mean_cost = np.mean(NN_costs[converged_mask]) if n_converged > 0 else np.nan
            
            stability = stability_rate * 100
            cost = mean_cost
            row[f"{pert_name} Stab."] = f"{stability:.0f}%"
            row[f"{pert_name} Cost"] = f"{cost:.2f}" if cost < 1e6 and not np.isnan(cost) else "—"
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)