import numpy as np
from typing import Dict, Any
from .kkt import project_weights_kkt

class MMUpdate:
    """Base class for MM updates."""
    def compute_matrices(self, X: np.ndarray, r: np.ndarray) -> Dict[str, Any]:
        """Precompute matrices needed for updates."""
        raise NotImplementedError
        
    def update(self, w: np.ndarray, **params) -> np.ndarray:
        """Single MM update step."""
        raise NotImplementedError

class ETEUpdate(MMUpdate):
    def compute_matrices(self, X: np.ndarray, r: np.ndarray) -> Dict[str, Any]:
        """Precompute matrices for ETE updates."""
        assert X.ndim == 2, "X must be 2-dimensional"
        assert X.shape[0] == r.shape[0], "X and r must have same number of samples"
        
        m = X.shape[0]
        A = (1/m) * X.T @ X
        Lmax_A = np.linalg.eigvalsh(A)[-1]
        
        return {
            'B': (2/Lmax_A) * (A - Lmax_A * np.eye(X.shape[1])),
            'b': (-2/m) * X.T @ r.flatten(),
            'Lmax_A': Lmax_A
        }
        
    def update(self, w: np.ndarray, B: np.ndarray, b: np.ndarray,
              Lmax_A: float, lambda_: float, p: float, c1: float, u: float) -> np.ndarray:
        """
        Perform single Majorization-Minimization update step for ETE objective.
        
        The MM update uses a quadratic majorizer of the original objective,
        leading to an update of the form:
            w+ = project_weights_kkt(Bw + (1/Lmax_A)(b + d))
        
        Args:
            w: Current weight vector of shape (n,)
            B: Precomputed matrix (2/Lmax_A)(A - Lmax_A*I) where A = (1/m)X'X
            b: Precomputed vector (-2/m)X'r
            Lmax_A: Largest eigenvalue of matrix A
            lambda_: Sparsity parameter
            p: Current value of smoothing parameter
            c1: log(1 + u/p), normalization constant for sparsity
            u: Upper bound on weights
        
        Returns:
            Updated weight vector satisfying constraints
        """
        assert w.ndim == 1, "w must be 1-dimensional"
        assert B.shape == (len(w), len(w)), "B dimensions must match w"
        assert len(b) == len(w), "b length must match w"

        d = lambda_ / ((p + np.abs(w)) * c1)  # derivative of log term
        c = B @ w + (1/Lmax_A) * (b + d)      # quadratic majorizer minimizer
        return project_weights_kkt(c, u)       # project onto feasible set

class DRUpdate(MMUpdate):
    def compute_matrices(self, X: np.ndarray, r: np.ndarray):
        """
        Precompute matrices needed for DR updates.
        """
        m = X.shape[0]
        A = (1/m) * X.T @ X
        Lmax_A = np.linalg.eigvalsh(A)[-1]
        B = (2/Lmax_A) * (A - Lmax_A * np.eye(X.shape[1]))
        b = (-2/m) * X.T @ r
        return {'B': B, 'b': b, 'Lmax_A': Lmax_A, 'X': X, 'r': r, 'm': m}
    
    def update(
        self, 
        w: np.ndarray, 
        B: np.ndarray, 
        b: np.ndarray,
        Lmax_A: float, 
        X: np.ndarray,
        r: np.ndarray,
        m: int,
        lambda_: float, 
        p: float, 
        c1: float, 
        u: float
    ) -> np.ndarray:
        """
        Single MM update step for DR measure.
        
        The update includes term for asymmetric penalization
        of underperformance.
        """
        
        h = np.minimum(r - X @ w, 0) # Compute negative part for downside risk
        d = lambda_ / ((p + np.abs(w)) * c1) # Derivative term from sparsity
        c = B @ w + (1/Lmax_A) * (b + d + 2/m * X.T @ h)
        return project_weights_kkt(c, u)

class HETEUpdate(MMUpdate):
    def compute_matrices(self, X: np.ndarray, r: np.ndarray):
        """Nothing to precompute for HETE as matrices change each iteration."""
        return {'X': X, 'r': r}
        
    def update(
        self,
        w: np.ndarray,
        X: np.ndarray,
        r: np.ndarray,
        lambda_: float,
        p: float,
        c1: float,
        hub: float,
        u: float,
        **kwargs
    ) -> np.ndarray:
        """
        Perform single MM update step for HETE measure.
        
        Uses quadratic majorizer based on Huber loss.
        """
        m = len(r)
        errors = r - X @ w
        
        # Compute weights for quadratic approximation
        alpha = np.ones(m)
        is_outlier = np.abs(errors) > hub
        alpha[is_outlier] = hub / np.abs(errors[is_outlier])
        
        # Construct weighted problem matrices
        Q = (1/m) * X.T @ np.diag(alpha) @ X
        Lmax = np.linalg.eigvalsh(Q)[-1]
        
        # Sparsity term
        d = lambda_ / ((p + np.abs(w)) * c1)
        
        # Combined update
        c = (1/Lmax) * (
            2*(Q - Lmax*np.eye(len(w))) @ w - 
            2/m * X.T @ (alpha * r) + 
            d
        )
        
        return project_weights_kkt(c, u)

class HDRUpdate(MMUpdate):
    def compute_matrices(self, X: np.ndarray, r: np.ndarray):
        """Nothing to precompute for HDR as matrices change each iteration."""
        return {'X': X, 'r': r}
        
    def update(self, w: np.ndarray, X: np.ndarray, r: np.ndarray,
               lambda_: float, p: float, c1: float, hub: float, u: float, **kwargs) -> np.ndarray:
        """Single MM update step for HDR measure."""
        m = len(r)
        tmp = r - X @ w
        
        alpha = np.ones(m)
        positive_and_below_hub = (tmp > 0) & (tmp <= hub)
        above_hub = tmp > hub
        
        if np.any(positive_and_below_hub):
            alpha[positive_and_below_hub] = 1.0  # quadratic region
        if np.any(above_hub):
            alpha[above_hub] = hub / tmp[above_hub]  # linear region
        
        Q = (1/m) * X.T @ np.diag(alpha) @ X
        Lmax = np.linalg.eigvalsh(Q)[-1]
        
        d = lambda_ / ((p + np.abs(w)) * c1)
        q = -np.maximum(X @ w - r, 0)
        
        c = (1/Lmax) * (
            2*(Q - Lmax*np.eye(len(w))) @ w + 
            2/m * X.T @ (alpha * (q - r)) + 
            d
        )
        
        return project_weights_kkt(c, u)