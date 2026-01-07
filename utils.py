import numpy as np

def u_analytic(Vx: np.ndarray, config) -> np.ndarray:
    """
    Compute optimal control from value gradient using PMP.
    
    Parameters:
    -----------
    Vx : array, shape (d,)
        Value gradient (costate)
    config : config-like object
        Must have: q (control penalty scalar or array), B (control matrix)
        
    Returns:
    --------
    u_optimal : array, shape (m,)
        Optimal control
    """
    q = config.q
    B = config.B

    m = B.shape[1]
    if np.isscalar(q):
        R_inv = np.eye(m) / (2.0 * q)
    else:
        R_inv = np.diag(1.0 / (2.0 * np.asarray(q).flatten()))
    u_optimal = -R_inv @ (B.T @ Vx)

    return u_optimal