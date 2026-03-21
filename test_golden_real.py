"""
Golden test: captures deterministic numerical outputs for the three kept systems
(burgers, allen_cahn, kuramoto_sivashinsky). Run with --save to create baseline,
then re-run after refactoring to verify no behavior change.
"""
import os, sys, json
import numpy as np

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

GOLDEN_FILE = os.path.join(_root, "golden_real.json")
SYSTEMS = ["burgers", "allen_cahn", "kuramoto_sivashinsky"]
ATOL = 1e-10


def compute_golden():
    from problems import create_config

    results = {}
    for system in SYSTEMS:
        config = create_config(system=system, seed=42)
        ocp = config.ocp

        np.random.seed(42)
        X = np.random.randn(config.n_states, 3) * 0.1
        U = np.random.randn(config.n_controls, 3) * 0.01

        F = ocp.dynamics(X, U)
        dFdX, dFdU = ocp.jacobians(X, U)
        L = ocp.running_cost(X, U)
        X0 = ocp.sample_initial_conditions(5, seed=42, K=10)
        U_lqr = ocp.LQR.eval_U(X)

        aug_dim = 2 * config.n_states + 1
        X_aug = np.random.randn(aug_dim, 3) * 0.1
        bvp_out = ocp.bvp_dynamics(0.0, X_aug)

        norm_vals = ocp.norm(X)

        results[system] = {
            "n_states": config.n_states,
            "n_controls": config.n_controls,
            "dynamics_sum": float(np.sum(F)),
            "dynamics_norm": float(np.linalg.norm(F)),
            "dFdX_sum": float(np.sum(dFdX)),
            "dFdU_sum": float(np.sum(dFdU)),
            "running_cost_sum": float(np.sum(L)),
            "X0_sum": float(np.sum(X0)),
            "X0_norm": float(np.linalg.norm(X0)),
            "U_lqr_sum": float(np.sum(U_lqr)),
            "U_lqr_norm": float(np.linalg.norm(U_lqr)),
            "bvp_sum": float(np.sum(bvp_out)),
            "bvp_norm": float(np.linalg.norm(bvp_out)),
            "norm_sum": float(np.sum(norm_vals)),
        }

    return results


def save_golden():
    results = compute_golden()
    with open(GOLDEN_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved golden results to {GOLDEN_FILE}")
    for sys_name, vals in results.items():
        print(f"  {sys_name}: {len(vals)} values")


def verify_golden():
    if not os.path.exists(GOLDEN_FILE):
        print(f"No golden file found at {GOLDEN_FILE}. Run with --save first.")
        sys.exit(1)

    with open(GOLDEN_FILE) as f:
        golden = json.load(f)

    current = compute_golden()
    all_ok = True

    for sys_name in SYSTEMS:
        g = golden[sys_name]
        c = current[sys_name]
        print(f"\n--- {sys_name} ---")
        for key in g:
            gv, cv = g[key], c[key]
            if isinstance(gv, (int,)):
                ok = gv == cv
            else:
                ok = abs(gv - cv) < ATOL
            status = "OK" if ok else "FAIL"
            if not ok:
                print(f"  {status} {key}: golden={gv}, current={cv}, diff={abs(gv-cv):.2e}")
                all_ok = False
            else:
                print(f"  {status} {key}")

    if all_ok:
        print("\nAll golden tests PASSED.")
    else:
        print("\nSome golden tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    if "--save" in sys.argv:
        save_golden()
    else:
        verify_golden()
