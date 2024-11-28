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
        tracking_error = np.linalg.norm(np.maximum(r - X @ w, 0)**2)
        sparsity_term = np.sum(np.log(1 + w/p))
        return (1/lambda_) * tracking_error + m/c1 * sparsity_term

class HETEObjective(TrackingObjective):
    def compute(
        self, 
        X: np.ndarray, 
        r: np.ndarray, 
        w: np.ndarray,
        lambda_: float, 
        p: float, 
        c1: float, 
        m: int,
        hub: float
    ) -> float:
        """
        Compute the Huber ETE objective value.
        
        Args:
            X: Asset returns (m x n)
            r: Index returns (m)
            w: Weights (n)
            lambda_: Sparsity parameter
            p: Smoothing parameter
            c1: Normalization constant log(1 + u/p)
            m: Number of samples
            hub: Huber parameter (transition point)
            
        Returns:
            Objective value combining Huber loss and sparsity
        """
        # Compute residuals
        errors = X @ w - r
        
        is_quadratic = np.abs(errors) <= hub
        huber_loss = np.zeros_like(errors)
        huber_loss[is_quadratic] = errors[is_quadratic]**2
        huber_loss[~is_quadratic] = hub * (2*np.abs(errors[~is_quadratic]) - hub)
        
        tracking_error = np.sum(huber_loss)
        sparsity_term = np.sum(np.log(1 + w/p))
        
        return (1/lambda_) * tracking_error + m/c1 * sparsity_term

class HDRObjective(TrackingObjective):
    def compute(
        self,
        X: np.ndarray,
        r: np.ndarray,
        w: np.ndarray,
        lambda_: float,
        p: float,
        c1: float,
        m: int,
        hub: float
    ) -> float:
        """
        Compute the Huber Downside Risk objective.
        
        Args:
            X: Asset returns (m x n)
            r: Index returns (m)
            w: Weights (n)
            lambda_: Sparsity parameter
            p: Smoothing parameter
            c1: Normalization constant log(1 + u/p)
            m: Number of samples
            hub: Huber parameter
            
        Returns:
            Objective value combining Huber loss and sparsity,
            only penalizing underperformance
        """
        errors = np.maximum(r - X @ w, 0)  # only care about underperformance
        
        # Apply Huber loss to downside errors
        is_quadratic = errors <= hub
        huber_loss = np.zeros_like(errors)
        huber_loss[is_quadratic] = errors[is_quadratic]**2
        huber_loss[~is_quadratic] = hub * (2*errors[~is_quadratic] - hub)
        
        tracking_error = np.sum(huber_loss)
        sparsity_term = np.sum(np.log(1 + w/p))
        
        return (1/lambda_) * tracking_error + m/c1 * sparsity_term