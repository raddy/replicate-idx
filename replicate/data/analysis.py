import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
from ..evaluation.metrics import TrackingMetrics
from ..evaluation.visualizer import TrackingVisualizer
from .loading import DataLoader
import polars as pl

def train_test_split_data(
    loader: DataLoader,
    data_type: str,
    target_symbol: str,
    train_pct: float = 0.7,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_null_pct: float = 0.01,
    time_col: str = 'Date'
) -> Dict[str, Dict]:
    """
    Load data and split into train/test sets with metrics.
    
    Args:
        loader: DataLoader instance
        data_type: Type of data to load (e.g., DataLoader.CRYPTO_RETURNS)
        target_symbol: Symbol to use as target (e.g., "BTC")
        train_pct: Percentage of data to use for training (default: 0.7)
        start_date: Optional start date filter (format: YYYY-MM-DD)
        end_date: Optional end date filter (format: YYYY-MM-DD)
        max_null_pct: Maximum percentage of nulls allowed (default: 0.01)
        time_col: Name of time column (default: 'Date')
        
    Returns:
        Dictionary containing train and test data:
        {
            'train': {
                'X': features array,
                'r': target array,
                'dates': dates array,
            },
            'test': {
                'X': features array,
                'r': target array,
                'dates': dates array,
            },
            'asset_names': List of asset names
        }
    """
    # Load data lazily
    target, constituents = loader.load_data(
        data_type=data_type,
        target_symbol=target_symbol,
        start_date=start_date,
        end_date=end_date,
        max_null_pct=max_null_pct
    )
    
    # Get total number of rows for splitting
    n_samples = target.select(pl.count()).collect().item()
    train_size = int(n_samples * train_pct)
    
    # Get asset names before any filtering
    asset_names = [c for c in constituents.collect_schema().names() if c != time_col]
    
    # Create train/test splits using row numbers
    with_row_nums = target.with_row_count("row_num")
    train_filter = pl.col("row_num") < train_size
    test_filter = pl.col("row_num") >= train_size
    
    # Split constituents (features)
    constituents_with_nums = constituents.with_row_count("row_num")
    train_X = constituents_with_nums.filter(train_filter).drop(["row_num", time_col])
    test_X = constituents_with_nums.filter(test_filter).drop(["row_num", time_col])
    
    # Split target
    train_r = with_row_nums.filter(train_filter).drop(["row_num", time_col])
    test_r = with_row_nums.filter(test_filter).drop(["row_num", time_col])
    
    # Split dates
    train_dates = with_row_nums.filter(train_filter).select(time_col)
    test_dates = with_row_nums.filter(test_filter).select(time_col)
    
    # Convert to numpy arrays
    train_X_np = train_X.collect().to_numpy()
    test_X_np = test_X.collect().to_numpy()
    train_r_np = train_r.collect().to_numpy().flatten()
    test_r_np = test_r.collect().to_numpy().flatten()
    train_dates_np = train_dates.collect().to_numpy().flatten()
    test_dates_np = test_dates.collect().to_numpy().flatten()
    
    # Validate no empty arrays
    if any(arr.size == 0 for arr in [train_X_np, test_X_np, train_r_np, test_r_np, train_dates_np, test_dates_np]):
        raise ValueError("One or more arrays are empty after train/test split")
    
    return {
        'train': {
            'X': train_X_np,
            'r': train_r_np,
            'dates': train_dates_np,
        },
        'test': {
            'X': test_X_np,
            'r': test_r_np,
            'dates': test_dates_np,
        },
        'asset_names': asset_names
    }

def analyze_train_test_results(
    train_data: Dict,
    test_data: Dict,
    weights: np.ndarray,
    asset_names: list,
    title_prefix: str = "",
    plot_combined: bool = True
) -> Tuple[Dict, Dict]:
    """
    Analyze and visualize train/test results using existing infrastructure.
    """
    # Validate data before calculations
    if 'X' not in train_data or 'r' not in train_data:
        raise ValueError("Train data missing required fields 'X' or 'r'")
    if 'X' not in test_data or 'r' not in test_data:
        raise ValueError("Test data missing required fields 'X' or 'r'")
    
    # Calculate portfolio returns
    train_port_returns = train_data['X'] @ weights
    test_port_returns = test_data['X'] @ weights
    
    # Additional validation
    if len(train_port_returns) == 0:
        raise ValueError("Empty train portfolio returns array")
    if len(test_port_returns) == 0:
        raise ValueError("Empty test portfolio returns array")
    
    # Calculate metrics
    train_metrics = TrackingMetrics.calculate_metrics(
        train_port_returns,
        train_data['r'],
        weights
    )
    
    test_metrics = TrackingMetrics.calculate_metrics(
        test_port_returns,
        test_data['r'],
        weights
    )
    
    if plot_combined:
        # Plot combined train/test visualization
        TrackingVisualizer.plot_train_test_analysis(
            train_dates=train_data['dates'],
            train_port_returns=train_port_returns,
            train_index_returns=train_data['r'],
            test_dates=test_data['dates'],
            test_port_returns=test_port_returns,
            test_index_returns=test_data['r'],
            weights=weights,
            asset_names=asset_names,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            title=title_prefix
        )
    else:
        # Plot separate train and test visualizations
        TrackingVisualizer.plot_tracking_analysis(
            train_data['dates'],
            train_port_returns,
            train_data['r'],
            weights,
            asset_names,
            metrics=train_metrics,
            title=f"{title_prefix}Training Period"
        )
        
        TrackingVisualizer.plot_tracking_analysis(
            test_data['dates'],
            test_port_returns,
            test_data['r'],
            weights,
            asset_names,
            metrics=test_metrics,
            title=f"{title_prefix}Testing Period"
        )
    
    return train_metrics, test_metrics