import time
import warnings
import numpy as np
from scipy.interpolate import interp1d
import sampling


import simulation

_headers = (
    '\n attempted |  solved   |  desired  ',
    '-----------------------------------'
)

n_candidates = 200 # burgers: 200

def generate(
        OCP, config, n_trajectories, controller=None, resolve_failed=True,
        verbose=0, suppress_warnings=True, adaptive_sampling="gradient"
    ):
    '''
    Generate data for an OCP by solving n_trajectories open loop OCPs. Uses LQR
    or a provided NN controller to warm start the BVP. Returns the portion of
    each trajectory up to the time taken for the running cost to approximately
    reach zero.

    Parameters
    ----------
    OCP : object
        Instance of QRnet.problem_template.TemplateOCP
    config : object
        Instance of QRnet.problem_template.MakeConfig
    n_trajectories: int
        Number of optimal trajectories to generate
    controller : object, optional
        Instantiated BaseNN subclass. If None (default), use LQR for warm start
    resolve_failed : bool, default=True
        If True, continue attempting to solve BVPs until get n_trajectories
        successful solutions
    verbose : int, default=0
        See scipy.integrate.solve_bvp
    suppress_warnings : bool, default=True
        If True, treat numpy warnings as BVP failures
    adaptive_sampling : str, default="gradient"
        Adaptive sampling method: "gradient" (gradient residual) or "hjb" (HJB residual norm)

    Returns
    -------
    data : dict
        Open loop optimal control data containing
        n_trajectories : int
            Number of successfully integrated BVPs
        t : (1, n_data) array
            Time instances of each data point
        X : (n_states, n_data) array
            Optimal states of each data point
        dVdX : (n_states, n_data) array
            Costates i.e. value gradient at each data point, if available
        V : (1, n_data) array
            Optimal cost at each state
        U : (n_controls, n_data) array
            Optimal control for each state
    n_attempt : int
        Number of attempted BVPs
    n_fail : int
        Number of failed solution attempts
    sol_time : float
        Total time of successful solution attempts in seconds
    fail_time : float
        Total time of failed solution attempts in seconds
    '''
    data = {}

    events = OCP.make_integration_events()

    def open_converged(X, U):
        return OCP.running_cost(X, U) < config.fp_tol

    # Eikonal (Section 3.2 & 4.3): uniform x0 in [-1,1] only. Others: adaptive sampling.
    sampling_controller = OCP.LQR

    X0_pool, _ = sampling.adaptive_sample_conditions(
        config, n_trajectories,
        controller=sampling_controller,
        n_candidates=n_candidates, # burgers: 200
        seed=config.seed
    )
    if X0_pool.shape[0] != OCP.n_states:
        X0_pool = X0_pool.T  # Ensure (n_states, n_trajectories)

    valid_mask = np.isfinite(X0_pool).all(axis=0)
    if not valid_mask.all():
        n_invalid = np.sum(~valid_mask)
        print(f"Warning: {n_invalid} non-finite initial conditions detected, filtering them out.")
        X0_pool = X0_pool[:, valid_mask]
    if X0_pool.shape[1] < n_trajectories:
        n_needed = n_trajectories - X0_pool.shape[1]
        if config.system == "eikonal":
            X0_additional = sampling.sample_conditions(config, n_needed, seed=None)
        else:
            X0_additional, _ = sampling.adaptive_sample_conditions(
                config, n_needed, controller=sampling_controller,
                n_candidates=n_candidates, seed=None
            )
        if X0_additional.shape[0] != OCP.n_states:
            X0_additional = X0_additional.T
        valid_additional = np.isfinite(X0_additional).all(axis=0)
        X0_additional = X0_additional[:, valid_additional]
        if X0_additional.shape[1] > 0:
            X0_pool = np.hstack([X0_pool, X0_additional])
    if X0_pool.shape[1] < n_trajectories:
        n_still_needed = n_trajectories - X0_pool.shape[1]
        X0_simple = sampling.sample_conditions(config, n_still_needed, seed=None)
        if X0_simple.shape[0] != OCP.n_states:
            X0_simple = X0_simple.T
        X0_pool = np.hstack([X0_pool, X0_simple[:, :n_still_needed]])

    n_attempt = 0
    n_sol = 0
    n_fail = 0
    sol_time = []
    fail_time = []

    if resolve_failed:
        n_track = lambda : n_sol
    else:
        n_track = lambda : n_attempt

    # ------------------------------------------------------------------------ #

    with warnings.catch_warnings():
        if suppress_warnings:
            np.seterr(over='warn', divide='warn', invalid='warn')
            warnings.filterwarnings('error', category=RuntimeWarning)
        warnings.filterwarnings('error', category=UserWarning)

        print('\nSolving open loop OCPs...')
        for header in _headers:
            print(header)
        w = str(len('attempted') + 2)
        row = '{att:^' + w + 'd}|{sol:^' + w + 'd}|{des:^' + w + 'd}'

        while n_track() < n_trajectories:
            # Ensure we have enough samples in the pool
            if n_track() >= X0_pool.shape[1]:
                # Need more samples - resample
                n_needed = n_trajectories - n_track()

                X0_additional, _ = sampling.adaptive_sample_conditions(
                    config, n_needed,
                    controller=sampling_controller,
                    n_candidates=n_candidates, # burgers: 5
                    seed=None
                )
                if X0_additional.shape[0] != OCP.n_states:
                    X0_additional = X0_additional.T
                # Filter non-finite values
                valid_additional = np.isfinite(X0_additional).all(axis=0)
                X0_additional = X0_additional[:, valid_additional]
                if X0_additional.shape[1] > 0:
                    X0_pool = np.hstack([X0_pool, X0_additional])
            
            X0 = X0_pool[:, n_track()].flatten()
            
            # Validate X0 before using it
            if not np.isfinite(X0).all():
                # Skip this invalid initial condition and resample
                if resolve_failed:
                    # Try to get a valid sample
                    max_resample_attempts = 10
                    X0_new = None
                    for _ in range(max_resample_attempts):
                        X0_candidate = sampling.sample_conditions(config, 1, seed=None)
                        if X0_candidate.shape[0] != OCP.n_states:
                            X0_candidate = X0_candidate.T
                        if np.isfinite(X0_candidate).all():
                            X0_new = X0_candidate.flatten()
                            break
                    
                    if X0_new is not None:
                        # Replace in pool and use the new X0
                        if n_track() < X0_pool.shape[1]:
                            X0_pool[:, n_track()] = X0_new
                        else:
                            X0_pool = np.hstack([X0_pool, X0_new.reshape(-1, 1)])
                        X0 = X0_new
                        # Continue with the resampled X0 (don't skip)
                    else:
                        # Still invalid after max attempts, skip this attempt
                        n_attempt += 1
                        n_fail += 1
                        fail_time.append(0.0)
                        continue
                else:
                    # Not resolving failed, just skip
                    n_attempt += 1
                    n_fail += 1
                    fail_time.append(0.0)
                    continue

            n_attempt += 1

            start_time = time.time()

            try:
                # Integrates the closed-loop system to warm start the OCP solver
                t, X, ode_converged = simulation.sim_to_converge(
                    OCP.dynamics, OCP.closed_loop_jacobian, controller, X0,
                    config, events=events
                )

                # Validate simulation results
                if not np.isfinite(X).all() or X.shape[1] == 0:
                    # Simulation produced invalid results, treat as failure
                    warnings.warn(UserWarning())
                    raise RuntimeWarning("Simulation produced non-finite values")
# Modified Logic
                if ode_converged:
                    V, dVdX, U = controller.bvp_guess(X)
                else:
                    # Still use the WarmStart physics for the guess! 
                    # Do NOT reset V and dVdX to zero.
                    t_raw = t # preserve the simulation time
                    V, dVdX, U = controller.bvp_guess(X)


                # Solves the two-point BVP until to convergence to infinite
                # horizon approximation
                ocp_sol, cont_ocp_sol, ocp_converged = simulation.solve_ocp(
                    OCP, config,
                    t_guess=t, X_guess=X, U_guess=U, dVdX_guess=dVdX, V_guess=V,
                    solve_to_converge=True,
                    verbose=verbose,  # Set this to 2
                    suppress_warnings=suppress_warnings
                )

                if ocp_converged:
                    sol_time.append(time.time() - start_time)

                    n_sol += 1

                    t = np.concatenate((ocp_sol['t'].flatten(), t.flatten()))
                    ocp_sol['t'] = np.unique(t)
                    ocp_sol.update(cont_ocp_sol(ocp_sol['t']))

                    # Clips the trajectory to when the running cost first gets
                    # close to zero, reducing the concentration of data near the
                    # equilibrium
                    keep_idx, _ = simulation.clip_trajectory(
                        ocp_sol['t'], ocp_sol['X'], ocp_sol['U'], open_converged
                    )

                    for key, new_data in ocp_sol.items():
                        if key not in data:
                            data[key] = []
                        data[key].append(np.atleast_2d(new_data)[:,:keep_idx+1])
                else:
                    # OCP failed to converge - count as failure
                    warnings.warn(UserWarning())
                    raise RuntimeWarning("OCP failed to converge")

            except (UserWarning, RuntimeWarning, ValueError) as e:
                fail_time.append(time.time() - start_time)

                n_fail += 1
                
                # Debug: print failure reason if verbose
                if verbose > 0:
                    print(f"\nFailure reason: {type(e).__name__}: {str(e)}")

                # Resample the failed initial condition
                if resolve_failed:
                    # Try to get a valid sample
                    max_resample_attempts = 10
                    X0_new = None
                    for _ in range(max_resample_attempts):
                        X0_candidate = sampling.sample_conditions(config, 1, seed=None)
                        if X0_candidate.shape[0] != OCP.n_states:
                            X0_candidate = X0_candidate.T
                        if np.isfinite(X0_candidate).all():
                            X0_new = X0_candidate.flatten()
                            break
                    
                    if X0_new is not None:
                        # Replace in pool at the current position
                        if n_sol < X0_pool.shape[1]:
                            X0_pool[:, n_sol] = X0_new
                        else:
                            # Need to extend pool
                            X0_pool = np.hstack([X0_pool, X0_new.reshape(-1, 1)])

            if verbose:
                for header in _headers:
                    print(header)
            print(
                row.format(att=n_attempt, sol=n_sol, des=n_trajectories),
                end='\r'
            )

    for key, val in data.items():
        data[key] = np.hstack(val)
    data['n_trajectories'] = n_sol

    sol_time, fail_time = np.sum(sol_time), np.sum(fail_time)

    return data, n_attempt, n_fail, sol_time, fail_time



import os, json, hashlib
import torch
import numpy as np

def _fingerprint(config, n_trajectories, controller, adaptive_sampling="gradient"):
    ctrl_name = controller.__class__.__name__ if controller is not None else "None"

    payload = dict(
        system=config.system,
        seed=config.seed,
        n_states=config.n_states,
        n_controls=config.n_controls,
        n_trajectories=int(n_trajectories),

        ocp_solver=config.ocp_solver,
        t1_initial=config.t1_initial,
        t1_scale=config.t1_scale,
        t1_max=config.t1_max,
        fp_tol=config.fp_tol,
        direct_n_init_nodes=config.direct_n_init_nodes,
        indirect_tol=config.indirect_tol,
        indirect_max_nodes=config.indirect_max_nodes,

        controller=ctrl_name,
        adaptive_sampling=adaptive_sampling,
    )
    s = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest(), payload




def load_or_generate(config, n_trajectories, *, controller=None, cache_dir=None, force_regen=False, val_split=0.2, adaptive_sampling="gradient", **kwargs):
    if cache_dir is None:
        from problems import get_results_dir
        cache_dir = get_results_dir(config, "data")
    os.makedirs(cache_dir, exist_ok=True)

    key, meta = _fingerprint(config, n_trajectories, controller, adaptive_sampling=adaptive_sampling)
    path_train = os.path.join(cache_dir, f"pmp_{key}_train.npz")
    path_val = os.path.join(cache_dir, f"pmp_{key}_val.npz")

    if (not force_regen) and os.path.exists(path_train) and os.path.exists(path_val):
        # Load both train and validation sets
        z_train = np.load(path_train, allow_pickle=True)
        z_val = np.load(path_val, allow_pickle=True)
        dataset_train = {k: z_train[k] for k in z_train.files if k != "__meta__"}
        dataset_val = {k: z_val[k] for k in z_val.files if k != "__meta__"}
        meta_loaded = json.loads(str(z_train["__meta__"]))
        return dataset_train, dataset_val, meta_loaded

    # Generate data
    dataset, n_attempt, n_fail, sol_time, fail_time = generate(
        config.ocp, config, n_trajectories,
        controller=controller,
        adaptive_sampling=adaptive_sampling,
        **kwargs
    )

    # Split into train and validation sets
    n_data = dataset['X'].shape[1] if 'X' in dataset else dataset.get('t', np.array([])).shape[0]
    if n_data == 0:
        raise ValueError("Generated dataset is empty")
    
    # Create random permutation for splitting
    np.random.seed(config.seed)
    indices = np.random.permutation(n_data)
    n_val = int(n_data * val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # Split the data
    dataset_train = {}
    dataset_val = {}
    for key, val in dataset.items():
        if key == 'n_trajectories':
            # Keep original count for metadata, but we'll update it
            dataset_train[key] = val
            dataset_val[key] = val
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                dataset_train[key] = val[train_indices]
                dataset_val[key] = val[val_indices]
            elif val.ndim == 2:
                dataset_train[key] = val[:, train_indices]
                dataset_val[key] = val[:, val_indices]
            else:
                # For higher dimensional arrays, assume last dimension is the data dimension
                dataset_train[key] = val[..., train_indices]
                dataset_val[key] = val[..., val_indices]
        else:
            # For non-array data, copy to both
            dataset_train[key] = val
            dataset_val[key] = val

    meta_out = dict(
        **meta,
        n_attempt=int(n_attempt),
        n_fail=int(n_fail),
        sol_time=float(sol_time),
        fail_time=float(fail_time),
        n_train=int(len(train_indices)),
        n_val=int(len(val_indices)),
        val_split=float(val_split),
    )

    # Save both train and validation sets
    np.savez_compressed(path_train, **dataset_train, __meta__=json.dumps(meta_out))
    np.savez_compressed(path_val, **dataset_val, __meta__=json.dumps(meta_out))
    
    return dataset_train, dataset_val, meta_out
