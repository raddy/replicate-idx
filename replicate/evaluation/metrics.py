import numpy as np
from typing import Dict

class TrackingMetrics:
    """Calculate and aggregate tracking performance metrics."""
    
    @staticmethod
    def calculate_metrics(
        portfolio_returns: np.ndarray,
        index_returns: np.ndarray,
        portfolio_weights: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive tracking metrics."""

        # Input validation
        assert(portfolio_returns.ndim == 1)
        assert(index_returns.ndim == 1)
        assert(portfolio_weights.ndim == 1)
        assert(portfolio_returns.shape[0] == index_returns.shape[0])
        
        # Basic tracking metrics
        tracking_error = np.sqrt(np.mean((portfolio_returns - index_returns)**2))
        correlation = np.corrcoef(portfolio_returns, index_returns)[0,1]
        
        # Cumulative returns
        cum_port = np.cumprod(1 + portfolio_returns) - 1
        cum_index = np.cumprod(1 + index_returns) - 1
        
        # Maximum deviation
        max_deviation = np.max(np.abs(cum_port - cum_index))
        
        # Risk metrics
        port_vol = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
        index_vol = np.std(index_returns) * np.sqrt(252)
        
        # Portfolio characteristics
        active_positions = np.sum(portfolio_weights > 1e-6)
        max_weight = np.max(portfolio_weights)
        
        return {
            'tracking_error': tracking_error,
            'correlation': correlation,
            'max_deviation': max_deviation,
            'portfolio_volatility': port_vol,
            'index_volatility': index_vol,
            'active_positions': active_positions,
            'maximum_weight': max_weight,
            'information_ratio': np.mean(portfolio_returns - index_returns) / tracking_error * np.sqrt(252)
        }
