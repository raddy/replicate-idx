import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict
import seaborn as sns
from matplotlib import rcParams
from datetime import datetime

class TrackingVisualizer:
    """Visualization tools for tracking analysis."""
    
    # Plot style constants
    FONT_SIZES = {
        'axis_label': 10,
        'title': 12,
        'tick_label': 9,
        'legend': 9,
        'weight_labels': 7
    }
    
    COLORS = {
        'index': 'black',
        'portfolio': '#8B0000',  # Dark red
        'tracking': '#808080',   # Grey
        'tracking_fill': '#D3D3D3'  # Light grey
    }
    
    @staticmethod
    def setup_style():
        """Set up publication-quality plot style."""
        sns.set_theme(style="whitegrid", font="serif")
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Computer Modern Roman']
        rcParams['text.usetex'] = True
        rcParams['axes.labelsize'] = TrackingVisualizer.FONT_SIZES['axis_label']
        rcParams['axes.titlesize'] = TrackingVisualizer.FONT_SIZES['title']
        rcParams['xtick.labelsize'] = TrackingVisualizer.FONT_SIZES['tick_label']
        rcParams['ytick.labelsize'] = TrackingVisualizer.FONT_SIZES['tick_label']
        rcParams['legend.fontsize'] = TrackingVisualizer.FONT_SIZES['legend']
        rcParams['figure.titlesize'] = TrackingVisualizer.FONT_SIZES['title']
    
    @staticmethod
    def _style_axis(ax, grid='y', spines=False):
        """Apply common axis styling."""
        if grid == 'y':
            ax.grid(True, axis='y', alpha=0.3)
        else:
            ax.grid(True, alpha=0.3)
        
        if not spines:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    @staticmethod
    def _plot_returns(ax, dates, cum_index, cum_port):
        """Plot cumulative returns comparison."""
        ax.plot(dates, cum_index * 100, label='Index', 
               color=TrackingVisualizer.COLORS['index'], linewidth=1.5)
        ax.plot(dates, cum_port * 100, label='Portfolio',
               color=TrackingVisualizer.COLORS['portfolio'], linewidth=1.5, linestyle='--')
        ax.set_title('Cumulative Returns (\%)')
        ax.legend(frameon=True, fancybox=True, facecolor='white', edgecolor='gray')
        TrackingVisualizer._style_axis(ax)
    
    @staticmethod
    def _plot_tracking(ax, dates, tracking_diff):
        """Plot tracking difference."""
        # Use single grey color with different alphas for fill
        ax.fill_between(dates, tracking_diff, 0, 
                       color=TrackingVisualizer.COLORS['tracking_fill'], alpha=0.3)
        ax.plot(dates, tracking_diff, 
               color=TrackingVisualizer.COLORS['tracking'], linewidth=1)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('Tracking Difference (bps)')
        TrackingVisualizer._style_axis(ax)
    
    @staticmethod
    def _plot_weights(ax, weights, asset_names=None):
        """Plot portfolio weight allocation."""
        if asset_names is None:
            asset_names = [f'Asset {i+1}' for i in range(len(weights))]
        
        active_mask = weights > 1e-6
        active_weights = weights[active_mask]
        active_names = [name for name, is_active in zip(asset_names, active_mask) 
                       if is_active]
        
        # Use a single color palette for weights
        ax.bar(range(len(active_weights)), active_weights * 100, 
               color=TrackingVisualizer.COLORS['portfolio'], alpha=0.7)
        ax.set_xticks(range(len(active_weights)))
        ax.set_xticklabels(active_names, rotation=40, ha='right', 
                          fontsize=TrackingVisualizer.FONT_SIZES['weight_labels'])
        ax.tick_params(axis='x', pad=5)
        ax.set_title('Portfolio Weights (\%)')
        TrackingVisualizer._style_axis(ax)
    
    @staticmethod
    def plot_tracking_analysis(
        dates: np.ndarray,
        portfolio_returns: np.ndarray,
        index_returns: np.ndarray,
        weights: np.ndarray,
        asset_names: Optional[List[str]] = None,
        title: str = "Tracking Analysis",
        metrics: Optional[Dict[str, float]] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 10)
    ):
        """
        Create tracking analysis plots.
        
        Args:
            dates: Array of dates for the time series
            portfolio_returns: Portfolio returns
            index_returns: Index returns
            weights: Portfolio weights
            asset_names: Names of assets (optional)
            title: Plot title
            metrics: Dictionary of tracking metrics to display
            save_path: Path to save figure (optional)
            figsize: Figure size in inches
        """
        TrackingVisualizer.setup_style()
        
        # Create figure with gridspec for flexible layout
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 2])
        
        # Cumulative returns
        ax1 = fig.add_subplot(gs[0, :])
        cum_port = np.cumprod(1 + portfolio_returns) - 1
        cum_index = np.cumprod(1 + index_returns) - 1
        TrackingVisualizer._plot_returns(ax1, dates, cum_index, cum_port)
        
        # Tracking difference
        ax2 = fig.add_subplot(gs[1, :])
        tracking_diff = (cum_port - cum_index) * 100
        TrackingVisualizer._plot_tracking(ax2, dates, tracking_diff)
        
        # Weight allocation
        ax3 = fig.add_subplot(gs[2, :3])
        TrackingVisualizer._plot_weights(ax3, weights, asset_names)
        
        # Metrics table
        if metrics:
            ax4 = fig.add_subplot(gs[2, 3])
            ax4.axis('off')
            metrics_text = '\n'.join([
                f'{k}: {v:.4f}' for k, v in metrics.items()
            ])
            ax4.text(0.1, 0.9, metrics_text, 
                    transform=ax4.transAxes,
                    verticalalignment='top',
                    fontsize=9,
                    family='monospace')
            ax4.set_title('Tracking Metrics')
        
        # Overall title
        fig.suptitle(title, y=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_weight_evolution(
        dates: np.ndarray,
        weight_history: np.ndarray,
        asset_names: Optional[List[str]] = None,
        title: str = "Weight Evolution",
        save_path: Optional[str] = None,
        figsize: tuple = (8, 6)
    ):
        """
        Plot the evolution of portfolio weights over time.
        
        Args:
            dates: Array of dates
            weight_history: Array of shape (n_dates, n_assets)
            asset_names: Names of assets
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size in inches
        """
        TrackingVisualizer.setup_style()
        
        if asset_names is None:
            asset_names = [f'Asset {i+1}' for i in range(weight_history.shape[1])]
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot stacked area chart
        ax.stackplot(dates, weight_history.T * 100,
                    labels=asset_names,
                    alpha=0.8)
        
        ax.set_title(title)
        ax.set_ylabel('Weight Allocation (\%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig