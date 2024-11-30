import numpy as np
from typing import Optional, Dict, Any, Literal
from .objectives import TrackingObjective
from .updates import MMUpdate

class MMOptimizer:
    """MM optimization framework for sparse index tracking."""
    
    def __init__(
        self,
        measure: Literal['ete', 'dr', 'hete', 'hdr'] = 'ete'
    ):
        """
        Initialize optimizer with specific measure.
        
        Args:
            measure: Type of tracking measure to use
                'ete': Empirical Tracking Error
                'dr': Downside Risk
                'hete': Huber Empirical Tracking Error
                'hdr': Huber Downside Risk
        """
        self.measure = measure
        
        # Select appropriate objective and update method based on measure
        if measure == 'ete':
            from .objectives import ETEObjective
            from .updates import ETEUpdate
            self.objective = ETEObjective()
            self.updater = ETEUpdate()
        elif measure == 'dr':
            from .objectives import DRObjective
            from .updates import DRUpdate
            self.objective = DRObjective()
            self.updater = DRUpdate()
        elif measure == 'hete':
            from .objectives import HETEObjective
            from .updates import HETEUpdate
            self.objective = HETEObjective()
            self.updater = HETEUpdate()
        elif measure == 'hdr':
            from .objectives import HDRObjective
            from .updates import HDRUpdate
            self.objective = HDRObjective()
            self.updater = HDRUpdate()
        else:
            raise ValueError(f"Unknown measure: {measure}")
        
    def optimize(
        self,
        X: np.ndarray,
        r: np.ndarray,
        lambda_: float,
        u: float = 1.0,
        p_neg_exp: float = 7,
        max_iter: int = 1000,
        thres: float = 1e-9,
        w0: Optional[np.ndarray] = None,
        hub: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Optimize portfolio weights using MM algorithm.
        
        Args:
            X: Asset returns matrix of shape (m, n)
            r: Target returns vector of shape (m,)
            lambda_: Sparsity parameter (suggested range [1e-8, 1e-6])
            u: Upper bound on weights (default: 1.0)
            p_neg_exp: Final negative exponent of p (default: 7)
            max_iter: Maximum number of iterations (default: 1000)
            thres: Threshold for setting weights to zero (default: 1e-9)
            w0: Initial weights (default: uniform)
            hub: Huber parameter for HETE/HDR measures (default: 1.0)
            
        Returns:
            Dictionary containing:
            - weights: Optimal portfolio weights
            - objective_values: History of objective values
            - iterations: Number of iterations performed
            - success: Whether optimization succeeded
        """
        m, n = X.shape

        if n * u < 1:
            raise ValueError(
                f"Constraints impossible to satisfy: with {n} assets and max weight {u}, "
                f"maximum possible weight sum is {n*u:.2f}. Either increase u to at least "
                f"{1/n:.3f} or provide more assets."
        )
        
        # Initialize weights if not provided
        if w0 is None:
            w0 = np.ones(n) / n
            
        # Setup decreasing p sequence
        K = 10  # number of outer iterations
        p1 = 1  # first value of -log(p)
        gamma = (p_neg_exp/p1)**(1/K)
        pp = p1 * gamma**np.arange(K+1)
        pp = 10**(-pp)
        
        tol = np.minimum(pp/10, 1e-3)
        
        # Precompute matrices for MM updates
        matrices = self.updater.compute_matrices(X, r)
        
        # Initialize tracking
        F_v = np.zeros(max_iter)
        k = 0  # iteration counter
        w = w0.copy()
        
        # Main loop over decreasing p values
        for ee, p in enumerate(pp):
            c1 = np.log(1 + u/p)
            flg = 1
            
            while True:
                if k >= max_iter:
                    return {
                        'weights': w,
                        'objective_values': F_v[:k],
                        'iterations': k,
                        'success': False
                    }
                    
                # Update weights
                w_old = w.copy()
                
                # Pass hub parameter for Huber methods
                extra_params = {'hub': hub} if self.measure in ['hete', 'hdr'] else {}
                
                # Compute objective
                F_v[k] = self.objective.compute(
                    X=X, r=r, w=w, lambda_=lambda_, 
                    p=p, c1=c1, m=m, **extra_params
                )
                
                # MM update with acceleration
                w1 = self.updater.update(
                    w=w,
                    lambda_=lambda_,
                    p=p,
                    c1=c1,
                    u=u,
                    **matrices,
                    **extra_params
                )
                
                w2 = self.updater.update(
                    w=w1,
                    lambda_=lambda_,
                    p=p,
                    c1=c1,
                    u=u,
                    **matrices,
                    **extra_params
                )
                
                # Acceleration
                R = w1 - w
                U = w2 - w1 - R
                a = max(min(-np.linalg.norm(R) / np.linalg.norm(U), -1), -300)
                
                # Backtracking
                while True:
                    if abs(a+1) < 1e-6:
                        w_new = w2
                        F_v[k] = self.objective.compute(
                            X=X,
                            r=r,
                            w=w_new,
                            lambda_=lambda_,
                            p=p,
                            c1=c1,
                            m=m,
                            **extra_params
                        )
                        w = w_new
                        break
                        
                    w_new = w - 2*a*R + a**2*U
                    from .kkt import project_weights_kkt  # assuming this exists
                    w_new = project_weights_kkt(-2*w_new, u)
                    F_v[k] = self.objective.compute(
                        X=X,
                        r=r,
                        w=w_new,
                        lambda_=lambda_,
                        p=p,
                        c1=c1,
                        m=m,
                        **extra_params
                    )
                    
                    if flg == 0 and F_v[k] * (1 - np.sign(F_v[k])*1e-9) >= F_v[max(k-1, 0)]:
                        a = (a-1)/2
                    else:
                        w = w_new
                        break
                
                k += 1
                
                # Check convergence
                if flg == 0:
                    rel_change = abs(F_v[k-1] - F_v[k-2]) / max(1, abs(F_v[k-2]))
                    if rel_change <= tol[ee]:
                        break
                        
                flg = 0
        
        # Threshold and normalize
        w[w < thres] = 0
        if np.sum(w) > 0:
            w = w / np.sum(w)
            
        return {
            'weights': w,
            'objective_values': F_v[:k],
            'iterations': k,
            'success': k < max_iter,
            'final_objective': F_v[k-1] if k > 0 else None
        }