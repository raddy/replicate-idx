import numpy as np
from typing import Protocol

class TrackingObjective(Protocol):
    """Protocol for tracking error objectives."""
    def compute(self, X: np.ndarray, r: np.ndarray, w: np.ndarray, **params) -> float:
        """Compute objective value."""
        ...

class ETEObjective:
    def compute(self, X: np.ndarray, r: np.ndarray, w: np.ndarray,
               lambda_: float, p: float, c1: float, m: int) -> float:
        """Compute ETE objective value."""
        assert X.shape == (m, len(w)), "X dimensions must match m and w"
        assert r.shape == (m,), "r must be m-dimensional"
        assert w.ndim == 1, "w must be 1-dimensional"
        assert lambda_ > 0, "lambda must be positive"
        assert p > 0, "p must be positive"
        assert c1 > 0, "c1 must be positive"
        
        tracking_error = np.linalg.norm(X @ w - r)**2
        sparsity_term = np.sum(np.log(1 + w/p))
        return (1/lambda_) * tracking_error + m/c1 * sparsity_term

class DRObjective(TrackingObjective):
    def compute(
        self, 
        X: np.ndarray, 
        r: np.ndarray, 
        w: np.ndarray,
        lambda_: float, 
        p: float, 
        c1: float, 
        m: int
    ) -> float:
        """
        Compute the Downside Risk (DR) objective value.
        The objective penalizes only underperformance:
        (1/λ)||max(r - Xw, 0)||₂² + (m/c₁)Σlog(1 + w/p)
        
        Args:
            X: Asset returns matrix (m x n)
            r: Target returns vector (m)
            w: Current weight vector (n)
            lambda_: Sparsity parameter
            p: Current value of smoothing parameter
            c1: Normalization constant log(1 + u/p)
            m: Number of samples
        """
        tracking_error = np.sum(np.maximum(r - X @ w, 0)**2)
        sparsity_term = np.sum(np.log(1 + w/p))
        return (1/lambda_) * tracking_error + m/c1 * sparsity_term
