import os
import sys

# Ensure project root is on path when script is run by full path (e.g. from IDE)
_script_dir = os.path.dirname(os.path.abspath(__file__))
# If script lives in a subdir (e.g. controls/), project root is parent
_project_root = os.path.dirname(_script_dir) if os.path.basename(_script_dir) == "controls" else _script_dir
if _project_root and _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import numpy as np
import random
import logging
from dataclasses import asdict

# --- IMPORTS FROM YOUR CODEBASE ---
try:
    from controls.train import TrainConfig
    from controls.model_factory import train_controllers_from_config
except ImportError as e:
    print("Error: Could not import project modules. Make sure you are in the project root.")
    print(f"  Project root added to path: {_project_root!r}")
    print(f"  Import error: {e}")
    sys.exit(1)

# --- USER CONFIGURATION: SETUP YOUR PROBLEM HERE ---
def get_problem_setup():
    """
    TODO: Instantiate your actual physics problem configuration here.
    We need 'config' (Problem Config) and some dummy 'data'.
    """
    
    # ---------------------------------------------------------
    # EXAMPLE (Uncomment and adapt to your specific problem):
    # from problems.burgers import BurgersConfig
    # config = BurgersConfig()
    # ---------------------------------------------------------
    
    # --- MOCK FALLBACK (Delete this if you load your real config above) ---
    # This mock attempts to satisfy the interface expected by GradNet/Train
    class MockLQR:
        def eval_dVdX(self, x): return np.zeros_like(x)
        def bvp_guess(self, x): return np.zeros(x.shape[1]), np.zeros_like(x), np.zeros(x.shape[1])

    class MockPhysics(torch.nn.Module):
        """Minimal physics module for unsupervised loss (dynamics, running_cost, get_control)."""
        def __init__(self, n=2, m=1):
            super().__init__()
            self.n, self.m = n, m
        def get_control(self, dVdX):
            return torch.zeros(dVdX.shape[0], self.m, device=dVdX.device, dtype=dVdX.dtype)
        def dynamics(self, x, u):
            return torch.zeros_like(x)
        def running_cost(self, x, u):
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    class MockOCP:
        X_bar = np.zeros(2)  # target equilibrium (n_states,)
        LQR = MockLQR()
        def sample_initial_conditions(self, n, seed=None, K=None):
            return np.random.randn(2, n)  # Assuming 2 states
        def physics_module(self):
            return MockPhysics(n=2, m=1)
            
    class MockConfig:
        system = "mock"
        seed = 42
        n_states = 2
        n_controls = 1
        ocp = MockOCP()
        def norm(self, x): return np.linalg.norm(x, axis=0)
        
    config = MockConfig()
    print("WARNING: Using MOCK problem config. Results are only structural, not physical.")
    # ---------------------------------------------------------

    # Create dummy data compatible with your data loading
    # Assuming data is a dict of arrays as seen in data.py
    n_train = 50
    d_state = config.n_states
    
    dummy_data = {
        't': np.linspace(0, 1, 100),
        'X': np.random.randn(d_state, n_train),  # Initial conditions
        'dVdX': np.random.randn(d_state, n_train),  # Required for val loader (costate labels)
    }
    
    return config, dummy_data

# --- TEST HARNESS ---

def set_seed(seed=42):
    """Make everything deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_short_experiment():
    """Runs a very short, deterministic training loop."""
    set_seed(42)
    
    config, data = get_problem_setup()
    
    # 1. Define a minimal TrainConfig (Fast execution)
    train_cfg = TrainConfig(
        sup_epochs=0,
        rollouts=5,
        batch_size=4,
        horizon=10,
        log_every=100,
        device="cpu"
    )

    # 2. Define a single controller to test
    # We test 'GradNet' as it likely touches most physics code
    controller_configs = {
        "Test_GradNet": {
            "enabled": True,
            "kind": "gradnet",
            "use_lqr": False,
            "train_mode": "unsupervised", 
        }
    }

    print("⚡ Starting micro-training run...")
    
    # 3. Run Training
    # We suppress stdout to keep the test clean
    with open(os.devnull, 'w') as f, np.errstate(all='ignore'):
         # Optional: redirect stdout if model_factory is very noisy
         # sys.stdout = f 
         results = train_controllers_from_config(
            controller_configs=controller_configs,
            config=config,
            train_cfg=train_cfg,
            data=data,
            val_data=data, # Use same data for val to simple things
            device="cpu",
            verbose=False
        )
         # sys.stdout = sys.__stdout__

    # 4. Extract Fingerprint
    # We want to save the final model weights and the final loss to ensure logic holds
    model_name = "Test_GradNet"
    model = results[0][model_name] # trained_models dict
    history = results[2][model_name] # histories dict
    
    # Get checksum of all parameters
    param_checksum = torch.cat([p.view(-1) for p in model.parameters()]).sum().item()
    
    # Get final loss (if available in history)
    # Assuming history is a dict of lists, grab the last value of the first key found
    final_loss = 0.0
    if history and isinstance(history, dict):
        first_key = list(history.keys())[0]
        if len(history[first_key]) > 0:
            final_loss = history[first_key][-1]

    return {
        "param_checksum": param_checksum,
        "final_loss": final_loss,
        "config_str": str(asdict(train_cfg)) # Ensure config defaults didn't shift
    }

def main():
    GOLDEN_FILE = "golden_results.pt"
    
    print(f"--- AdHoc Safety Net ---")
    current_result = run_short_experiment()
    print(f"Current Result: Checksum={current_result['param_checksum']:.6f}, Loss={current_result['final_loss']:.6f}")

    if not os.path.exists(GOLDEN_FILE):
        print(f"\n⚠️  No golden file found. Saving current results as the GOLDEN STANDARD.")
        print(f"    File saved to: {GOLDEN_FILE}")
        torch.save(current_result, GOLDEN_FILE)
    else:
        golden_result = torch.load(GOLDEN_FILE, weights_only=False)
        
        # Compare
        match = True
        for k, v in golden_result.items():
            curr_v = current_result.get(k)
            
            # Float comparison with tolerance
            if isinstance(v, float):
                if not np.isclose(v, curr_v, atol=1e-6):
                    print(f"❌ MISMATCH in {k}: Golden={v}, Current={curr_v}")
                    match = False
            elif v != curr_v:
                print(f"❌ MISMATCH in {k}: Golden={v}, Current={curr_v}")
                match = False
                
        if match:
            print("\n✅ SUCCESS: Current code matches Golden Standard perfectly.")
        else:
            print("\n🚨 FAILURE: Logic has changed! Revert changes or delete 'golden_results.pt' if this was intentional.")
            sys.exit(1)

if __name__ == "__main__":
    main()