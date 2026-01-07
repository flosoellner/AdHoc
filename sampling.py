import numpy as np

def sample_conditions(config, n: int, seed: int = None, dist: float = None):
    """
    Sample initial conditions using sum of sine functions.
    
    Parameters:
    -----------
    config : MakeOCP instance or config-like object
        Must have: n_states (or d), xi (Chebyshev nodes), w (weights for norm)
    n : int
        Number of samples to generate
    seed : int, optional
        Random seed for reproducibility
    dist : float, optional
        If provided, normalize samples to this distance from origin
        
    Returns:
    --------
    X0 : array, shape (n, d)
        Initial conditions, one per row
    """
    if seed is not None:
        np.random.seed(seed)


    
    K = 10 # number of sine modes
    d = int(config.n_states)
    
    # Get xi (Chebyshev nodes) - may be on ocp or config
    xi = config.xi

    xi = xi.flatten()
    xi_pi = np.pi * xi  # Use pi * xi for sine functions
    

    # Generate samples: X0 = sum_{k=1}^10 a_k * sin(k * pi * xi), a_k ~ U[-1/k, 1/k]
    X0 = np.zeros((d, n))
    for k in range(1, K + 1):
        ak = (2.0 * np.random.rand(1, n) - 1.0) / float(k)
        X0 += ak * np.sin(k * xi_pi).reshape(d, 1)
    
    # Normalize to dist if requested
    if dist is not None:
        norm_func = config.norm

        X0_norm = norm_func(X0).reshape(1, -1)
        X0 *= float(dist) / (X0_norm + 1e-12)

    
    # Return as (n, d)
    return X0