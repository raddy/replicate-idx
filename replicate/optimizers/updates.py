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
