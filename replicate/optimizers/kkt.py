import numpy as np
from typing import Tuple, Optional

def project_weights_kkt(c: np.ndarray, u: float) -> np.ndarray:
    """
    Project onto simplex with upper bounds using KKT conditions.
    
    Args:
        c: Cost vector
        u: Upper bound on weights (must be in (0,1])
        
    Returns:
        Weights satisfying:
        - sum(w) = 1 (or maximum possible if u*n < 1)
        - 0 <= w <= u
    """
    assert c.ndim == 1, "c must be 1-dimensional"
    assert 0 < u <= 1, "u must be in (0,1]"
    
    tol = 1e-10
    n = len(c)
    sort_ind = np.argsort(c)
    c_sort = c[sort_ind]
    
    # Phase 1: Find number of active weights
    mid = _binary_search_active(c_sort, n)
    mu = -1/mid * (np.sum(c_sort[:mid]) + 2)
    
    # Check if solution satisfies upper bound
    w = np.zeros(n)
    weights = -(mu + c_sort[:mid])/2
    if np.all(weights <= u + tol):
        w[sort_ind[:mid]] = np.minimum(weights, u)
        return w
    
    # Phase 2: Binary search for split between u-weights and interior weights
    k = mid
    while True:
        result = _find_split_point(c_sort, k, u)
        if result is not None:
            split_point, mu = result
            w = np.zeros(n)
            w[sort_ind[:split_point]] = u
            w[sort_ind[split_point:k]] = -(mu + c_sort[split_point:k])/2
            if abs(np.sum(w) - 1.0) < tol:
                w[sort_ind[0]] += 1.0 - np.sum(w)  # Adjust first weight to ensure sum is exactly 1
                return w
            
        if k == n:
            # TODO -- can someone smarter than me think of a better way to do this?
            num_u_weights = int(np.ceil(1/u))
            w[:] = 0.0
            w[sort_ind[:min(num_u_weights, n)]] = u
            return w
        k += 1

def _binary_search_active(c_sort: np.ndarray, n: int) -> int:
    """Find number of active weights without upper bound."""
    high, low = n, 1
    while low <= high:
        mid = (low + high) // 2
        mu = -1/mid * (np.sum(c_sort[:mid]) + 2)
        
        if (_is_valid_partition(mu, c_sort, mid, n)):
            return mid
        elif mu + c_sort[mid-1] < 0:
            low = mid + 1
        else:
            high = mid - 1
    return mid

def _is_valid_partition(mu: float, c_sort: np.ndarray, mid: int, n: int) -> bool:
    """Check if partition is valid under KKT conditions."""
    return (mu + c_sort[mid-1] < 0 and 
            (mid == n or mu + c_sort[mid] >= 0))

def _find_split_point(c_sort: np.ndarray, k: int, u: float) -> Optional[Tuple[int, float]]:
    """
    Binary search for split between u-weights and interior weights.
    
    Args:
        c_sort: Sorted cost vector
        k: Current number of active weights
        u: Upper bound on weights
        
    Returns:
        Tuple of (split_point, mu) if found, None otherwise
    """
    low, high = 0, k - 1
    tol = 1e-10
    
    while low <= high:
        mid = (low + high) // 2
        mu = (2*mid*u - np.sum(c_sort[mid:k]) - 2) / (k - mid)
        
        # Check KKT conditions
        valid_u_weights = True if mid == 0 else (mu + c_sort[mid-1] <= -2*u + tol)
        valid_interior = (-2*u - tol < mu + c_sort[mid]) and (mu + c_sort[k-1] < tol)
        valid_inactive = k == len(c_sort) or mu + c_sort[k] >= -tol
        
        if valid_u_weights and valid_interior and valid_inactive:
            return mid, mu
        elif valid_u_weights and not valid_interior:
            low = mid + 1
        else:
            high = mid - 1
    
    return None

# Example to demonstrate phases:
def demonstrate_kkt_projection():
    # Case 1: No weights hit upper bound
    c1 = np.array([0.1, 0.2, 0.3, 0.4])
    u1 = 0.5
    w1 = project_weights_kkt(c1, u1)
    print("Case 1 (no upper bound hits):")
    print("Input c:", c1)
    print("Weights:", w1)
    print("Sum of weights:", np.sum(w1))
    print()
    
    # Case 2: Some weights hit upper bound
    c2 = np.array([-0.5, -0.4, 0.1, 0.2])
    u2 = 0.4
    w2 = project_weights_kkt(c2, u2)
    print("Case 2 (with upper bound hits):")
    print("Input c:", c2)
    print("Weights:", w2)
    print("Sum of weights:", np.sum(w2))
    print("Max weight:", np.max(w2))

    # Case 3: Too small u
    c = np.array([0.1, 0.2, 0.3])
    u = 0.1
    w = project_weights_kkt(c, u)
    print("Case 3 (too small u):")
    print("Input c:", c)
    print("Weights:", w)
    print("Sum of weights:", np.sum(w))

    # Case 4: Large values
    c = np.array([1e5, 1e6, 1e7])
    u = 0.5
    w = project_weights_kkt(c, u)
    print("Case 4 (large values):")
    print("Input c:", c)
    print("Weights:", w)
    print("Sum of weights:", np.sum(w))