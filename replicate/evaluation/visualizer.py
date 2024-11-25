import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

class TrackingVisualizer:
    """Visualization tools for tracking analysis."""
    
    @staticmethod
    def plot_tracking_analysis(
        dates: np.ndarray,
        portfolio_returns: np.ndarray,
        index_returns: np.ndarray,
        weights: np.ndarray,
        asset_names: Optional[List[str]] = None,
        title: str = "Tracking Analysis"
    ):
        """Create comprehensive tracking analysis plots."""
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Cumulative returns
        cum_port = np.cumprod(1 + portfolio_returns) - 1
        cum_index = np.cumprod(1 + index_returns) - 1
        
        axes[0].plot(dates, cum_index, label='Index', linewidth=2)
        axes[0].plot(dates, cum_port, label='Portfolio', linewidth=2, linestyle='--')
        axes[0].set_title('Cumulative Returns')
        axes[0].legend()
        axes[0].grid(True)
        
        # Tracking difference
        axes[1].plot(dates, cum_port - cum_index, color='red')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_title('Tracking Difference')
        axes[1].grid(True)
        
        # Weight allocation
        if asset_names is None:
            asset_names = [f'Asset {i+1}' for i in range(len(weights))]
            
        active_weights = weights[weights > 1e-6]
        active_names = [name for name, w in zip(asset_names, weights) if w > 1e-6]
        
        axes[2].bar(active_names, active_weights)
        axes[2].set_title('Portfolio Weights')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()