import polars as pl
from pathlib import Path
from typing import Optional, Tuple, Union, Literal, List, Dict
from .paths import DataPaths

class DataLoader:
    """Data loading utilities for financial time series."""
    
    # Data types
    CRYPTO_RETURNS = "CRYPTO_RETURNS"
    SP500_2010 = "SP500_2010"
    SP500_2010_2015 = "SP500_2010_2015"
    
    def __init__(self):
        """Initialize DataLoader."""
        self.time_col_map = {
            self.CRYPTO_RETURNS: 'timestamp',
            self.SP500_2010: 'Date',
            self.SP500_2010_2015: 'Date'
        }
    
    def _apply_date_filter(
        self,
        lf: pl.LazyFrame,
        time_col: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.LazyFrame:
        """Apply date range filter to LazyFrame."""
        if not (start_date or end_date):
            return lf
            
        date_filter = []
        if start_date:
            date_filter.append(pl.col(time_col).cast(pl.Date) >= pl.lit(start_date).cast(pl.Date))
        if end_date:
            date_filter.append(pl.col(time_col).cast(pl.Date) <= pl.lit(end_date).cast(pl.Date))
        
        return lf.filter(pl.all_horizontal(date_filter))
    
    def _load_crypto(
        self,
        target_symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_null_pct: float = 0.01,
        **kwargs
    ) -> Union[Tuple[pl.LazyFrame, pl.LazyFrame], pl.LazyFrame]:
        """Load cryptocurrency return data."""
        # Load long-format data lazily
        df = pl.scan_parquet(DataPaths.CRYPTO_2024_LONG_PATH)

        time_col = self.time_col_map[self.CRYPTO_RETURNS]
        
        # Filter by date range using existing method
        df_filtered = self._apply_date_filter(df, time_col, start_date, end_date)
        
        # Calculate null percentages and filter symbols lazily
        symbol_stats = (
            df_filtered
            .group_by('symbol')
            .agg([
                pl.col('pct_return').null_count().alias('null_count'),
                pl.count().alias('total_count')
            ])
            .with_columns([
                (pl.col('null_count') / pl.col('total_count')).alias('null_pct')
            ])
        )

        # Only collect when getting valid symbols
        valid_symbols = symbol_stats.filter(pl.col('null_pct') <= max_null_pct).collect().get_column('symbol').to_list()
        
        # Process data lazily
        df_filtered = (
            df_filtered
            .filter(pl.col('symbol').is_in(valid_symbols))
            .drop('price')
            .select([time_col, 'symbol', 'pct_return'])
        )
        
        # Pivot to wide format and fill nulls
        wide_df = (
            df_filtered
            .collect()  # Need to collect for pivot
            .pivot(
                values='pct_return',
                index=time_col,
                columns='symbol'
            )
            .sort(time_col)
            .fill_null(0.0)
        )
        
        # Convert back to LazyFrame
        wide_lf = wide_df.lazy()
        
        if target_symbol:
            if target_symbol not in valid_symbols:
                raise ValueError(
                    f"Target symbol '{target_symbol}' not found in data or has too many null values "
                    f"(max allowed: {max_null_pct:.1%})"
                )
            # Split into target and constituents
            target_lf = wide_lf.select([time_col, target_symbol])
            constituents_lf = wide_lf.select(
                [time_col] + 
                [col for col in wide_df.columns if col not in [time_col, target_symbol]]
            )
            return target_lf, constituents_lf
        
        return wide_lf
    
    def _load_sp500(
        self,
        data_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """Load SP500 return and feature data."""
        # Map data types to paths
        path_map = {
            self.SP500_2010: (DataPaths.SP500_2010_R_PATH, DataPaths.SP500_2010_X_PATH),
            self.SP500_2010_2015: (DataPaths.SP500_2010_2015_R_PATH, DataPaths.SP500_2010_2015_X_PATH)
        }

        time_col = self.time_col_map[data_type]
        
        if data_type not in path_map:
            raise ValueError(f"Unknown SP500 data type: {data_type}")
            
        returns_path, features_path = path_map[data_type]
        
        # Load data
        returns_lf = pl.scan_parquet(returns_path)
        features_lf = pl.scan_parquet(features_path)
        
        # Apply date filters
        returns_lf = self._apply_date_filter(returns_lf, time_col, start_date, end_date)
        features_lf = self._apply_date_filter(features_lf, time_col, start_date, end_date)
        
        return features_lf, returns_lf
    
    def load(
        self,
        data_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        target_symbol: Optional[str] = None,
        **kwargs
    ) -> Union[Tuple[pl.LazyFrame, pl.LazyFrame], pl.LazyFrame]:
        """
        Load financial data.
        
        Args:
            data_type: Type of data to load (CRYPTO_RETURNS, SP500_2010, SP500_2010_2015)
            start_date: Start date for data range (format: 'YYYY-MM-DD')
            end_date: End date for data range (format: 'YYYY-MM-DD')
            target_symbol: Symbol to use as target for crypto data
            **kwargs: Additional arguments passed to specific loaders
            
        Returns:
            For SP500: Tuple of (returns_lf, features_lf) as LazyFrames
            For crypto: Single LazyFrame or tuple of (target_lf, constituents_lf) if target_symbol specified
            
        Raises:
            ValueError: If data_type is unknown or if validation fails
        """

        if data_type == self.CRYPTO_RETURNS:
            return self._load_crypto(
                target_symbol=target_symbol,
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )
        elif data_type in [self.SP500_2010, self.SP500_2010_2015]:
            return self._load_sp500(
                data_type=data_type,
                start_date=start_date,
                end_date=end_date,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unknown data type: {data_type}. "
                f"Available types: {[self.CRYPTO_RETURNS, self.SP500_2010, self.SP500_2010_2015]}"
            )