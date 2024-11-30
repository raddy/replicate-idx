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
        'tracking_fill': '#D3D3D3',  # Light grey
        'train_period': '#E8E8E8',  # Light grey for train period background
        'test_period': '#FFF0F0'   # Light red for test period background
    }
    
    @staticmethod
    def setup_style():
        """Set up publication-quality plot style."""
        sns.set_theme(style="whitegrid", font="serif")
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Computer Modern Roman']
        rcParams['text.usetex'] = False # Change back
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
    def _plot_period_background(ax, train_dates, test_dates):
        """Add train/test period background shading and labels."""
        # Add background colors
        ax.axvspan(train_dates[0], train_dates[-1], 
                  color=TrackingVisualizer.COLORS['train_period'], alpha=0.3)
        ax.axvspan(test_dates[0], test_dates[-1], 
                  color=TrackingVisualizer.COLORS['test_period'], alpha=0.3)
        
        # Add split line
        ax.axvline(x=train_dates[-1], color='black', linestyle=':', alpha=0.5)
        
        # Add period labels
        mid_train = train_dates[len(train_dates)//2]
        mid_test = test_dates[len(test_dates)//2]
        y_pos = ax.get_ylim()[1] * 0.95
        ax.text(mid_train, y_pos, 'Training', 
               horizontalalignment='center', verticalalignment='bottom')
        ax.text(mid_test, y_pos, 'Testing', 
               horizontalalignment='center', verticalalignment='bottom')

    @staticmethod
    def _plot_cumulative_returns(ax, dates, cum_port, cum_index, title=""):
        """Plot cumulative returns comparison."""
        ax.plot(dates, cum_index * 100, label='Index',
               color=TrackingVisualizer.COLORS['index'], linewidth=1.5)
        ax.plot(dates, cum_port * 100, label='Portfolio',
               color=TrackingVisualizer.COLORS['portfolio'], linewidth=1.5, linestyle='--')
        ax.set_title(f'{title}\nCumulative Returns (\%)')
        ax.legend(frameon=True, fancybox=True, facecolor='white', edgecolor='gray')
        TrackingVisualizer._style_axis(ax)

    @staticmethod
    def _plot_tracking_difference(ax, dates, tracking_diff):
        """Plot tracking difference."""
        ax.fill_between(dates, tracking_diff, 0,
                       color=TrackingVisualizer.COLORS['tracking_fill'], alpha=0.3)
        ax.plot(dates, tracking_diff,
               color=TrackingVisualizer.COLORS['tracking'], linewidth=1)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('Tracking Difference (\%)')
        TrackingVisualizer._style_axis(ax)

    @staticmethod
    def _format_metrics(metrics: Dict) -> str:
        """Format metrics dictionary into a readable string."""
        return (
            f"TE:      {metrics['tracking_error']*100:>6.2f}%\n"
            f"IR:      {metrics['information_ratio']:>6.2f}\n"
            f"Corr:    {metrics['correlation']:>6.3f}\n"
            f"MaxDev:  {metrics['max_deviation']*100:>6.2f}%\n"
            f"PortVol: {metrics['portfolio_volatility']*100:>6.2f}%\n"
            f"IdxVol:  {metrics['index_volatility']*100:>6.2f}%\n"
            f"ActPos:  {metrics['active_positions']:>6.0f}\n"
            f"MaxWgt:  {metrics['maximum_weight']*100:>6.1f}%"
        )

    @staticmethod
    def _plot_metrics(ax, train_metrics: Dict, test_metrics: Dict):
        """Plot metrics in a nicely formatted table."""
        ax.axis('off')
        
        # Set title with padding
        ax.set_title('Performance Metrics', pad=15)
        
        # Format metrics text
        train_text = TrackingVisualizer._format_metrics(train_metrics)
        test_text = TrackingVisualizer._format_metrics(test_metrics)
        
        # Add column headers with proper spacing
        ax.text(0.05, 0.90, "Training", fontweight='bold', fontsize=9,
                horizontalalignment='left', transform=ax.transAxes)
        ax.text(0.55, 0.90, "Testing", fontweight='bold', fontsize=9,
                horizontalalignment='left', transform=ax.transAxes)
        
        # Add metrics text with proper alignment
        ax.text(0.05, 0.80, train_text, fontsize=8, family='monospace',
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes)
        ax.text(0.55, 0.80, test_text, fontsize=8, family='monospace',
                verticalalignment='top', horizontalalignment='left',
                transform=ax.transAxes)

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
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    @staticmethod
    def plot_train_test_analysis(
        train_dates: np.ndarray,
        train_port_returns: np.ndarray,
        train_index_returns: np.ndarray,
        test_dates: np.ndarray,
        test_port_returns: np.ndarray,
        test_index_returns: np.ndarray,
        weights: np.ndarray,
        asset_names: list,
        train_metrics: Optional[Dict] = None,
        test_metrics: Optional[Dict] = None,
        title: str = "",
        figsize: tuple = (12, 8)
    ) -> None:
        """
        Plot full tracking analysis with train/test split visualization.
        
        Args:
            train_dates: Dates for training period
            train_port_returns: Portfolio returns in training period
            train_index_returns: Index returns in training period
            test_dates: Dates for testing period
            test_port_returns: Portfolio returns in testing period
            test_index_returns: Index returns in testing period
            weights: Portfolio weights
            asset_names: Names of assets
            train_metrics: Optional metrics for training period
            test_metrics: Optional metrics for testing period
            title: Optional plot title
            figsize: Figure size (width, height)
        """
        TrackingVisualizer.setup_style()
        
        # Create figure with gridspec
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1])
        
        # Create main axes
        ax1 = fig.add_subplot(gs[0, :])  # Cumulative returns
        ax2 = fig.add_subplot(gs[1, :])  # Tracking difference
        ax3 = fig.add_subplot(gs[2, :3])  # Weights
        ax4 = fig.add_subplot(gs[2, 3])   # Metrics
        
        # Combine dates and returns for full series
        all_dates = np.concatenate([train_dates, test_dates])
        all_port_returns = np.concatenate([train_port_returns, test_port_returns])
        all_index_returns = np.concatenate([train_index_returns, test_index_returns])
        
        # Calculate cumulative returns
        cum_port = np.cumprod(1 + all_port_returns) - 1
        cum_index = np.cumprod(1 + all_index_returns) - 1
        
        # Plot each component
        TrackingVisualizer._plot_period_background(ax1, train_dates, test_dates)
        TrackingVisualizer._plot_cumulative_returns(ax1, all_dates, cum_port, cum_index, title)
        
        TrackingVisualizer._plot_period_background(ax2, train_dates, test_dates)
        tracking_diff = (cum_port - cum_index) * 100
        TrackingVisualizer._plot_tracking_difference(ax2, all_dates, tracking_diff)
        
        TrackingVisualizer._plot_weights(ax3, weights, asset_names)
        
        # Add metrics if provided
        if train_metrics and test_metrics:
            TrackingVisualizer._plot_metrics(ax4, train_metrics, test_metrics)