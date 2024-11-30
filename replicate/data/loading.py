import polars as pl
from pathlib import Path
from typing import Optional, Tuple, Union, Literal, List, Dict
from .paths import DataPaths

class DataLoader:
    """Data loading utilities for financial time series."""
    
    # Standard data types
    SP500_INDEX = "SP500_INDEX"
    SP500_CONSTITUENTS = "SP500_CONSTITUENTS"
    CRYPTO_RETURNS = "CRYPTO_RETURNS"
    SP500_2010_2015 = "SP500_2010_2015"  # New data type for the 2010-2015 SP500 dataset
    
    def __init__(self, date_col: str = 'Date'):
        """
        Initialize DataLoader.
        
        Args:
            date_col: Default name of date column
        """
        self.date_col = date_col
    
    def load(
        self,
        path: Union[str, Path],
        parse_dates: bool = True,
        return_col: Optional[str] = None,
        validate: bool = True
    ) -> Tuple[pl.LazyFrame, Optional[str]]:
        """
        Lazily load return data from parquet file.
        
        Args:
            path: Path to parquet file
            parse_dates: Whether to parse dates from strings
            return_col: If specified, only this return column will be kept
            validate: Whether to validate the loaded data
            
        Returns:
            LazyFrame with date column first, followed by all return columns
                (or just the specified return column if return_col is provided)
            Name of return column used (only if return_col was specified)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If validation fails
        """
        # Input validation
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Load data
        lf = pl.scan_parquet(path)
        
        # Get column names once to avoid multiple schema resolutions
        columns = lf.collect_schema().names()
        
        # Validate columns
        if validate:
            if self.date_col not in columns:
                raise ValueError(f"Date column '{self.date_col}' not found")
            if return_col and return_col not in columns:
                raise ValueError(f"Return column '{return_col}' not found")
        
        # Select columns
        if return_col:
            lf = lf.select([self.date_col, return_col])
        else:
            cols = [self.date_col] + [c for c in columns if c != self.date_col]
            lf = lf.select(cols)
        
        # Parse dates if requested
        if parse_dates:
            lf = lf.with_columns(pl.col(self.date_col).str.strptime(pl.Date, format=None))
        
        # Validate data
        if validate:
            df = lf.collect()
            if df.is_empty():
                raise ValueError("Empty dataset")
            # Check for null values by summing null counts across all columns
            total_nulls = sum(df.null_count().row(0))
            if total_nulls > 0:
                raise ValueError("Dataset contains null values")
        
        return lf, return_col
    
    def get_crypto_symbol_availability(
        self,
        df: pl.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_null_pct: float = 0.01
    ) -> Tuple[List[str], Dict[str, Tuple[str, str]]]:
        """
        Get available crypto symbols and their availability dates.
        
        Args:
            df: DataFrame containing crypto data in long format
            start_date: Start date cutoff (format: 'YYYY-MM-DD')
            end_date: End date cutoff (format: 'YYYY-MM-DD')
            max_null_pct: Maximum percentage of null values allowed for a symbol (0.01 = 1%)
            
        Returns:
            Tuple of:
                - List of valid symbols (symbols with sufficient data in date range)
                - Dictionary mapping symbols to tuple of (first_date, last_date)
        """
        # Filter to date range if specified
        if start_date or end_date:
            date_filter = []
            if start_date:
                date_filter.append(pl.col('timestamp').cast(pl.Date) >= pl.lit(start_date).cast(pl.Date))
            if end_date:
                date_filter.append(pl.col('timestamp').cast(pl.Date) <= pl.lit(end_date).cast(pl.Date))
            df = df.filter(pl.all_horizontal(date_filter))
        
        # Calculate null percentages per symbol
        symbol_stats = (
            df.group_by('symbol')
            .agg([
                pl.col('timestamp').cast(pl.Date).min().alias('first_date'),
                pl.col('timestamp').cast(pl.Date).max().alias('last_date'),
                pl.col('pct_return').null_count().alias('null_count'),
                pl.count().alias('total_count')
            ])
            .with_columns([
                (pl.col('null_count') / pl.col('total_count')).alias('null_pct')
            ])
            .sort('first_date')
        )
        
        # Filter to symbols with acceptable null percentages
        valid_symbols_df = symbol_stats.filter(pl.col('null_pct') <= max_null_pct)
        
        # Create availability mapping
        availability = dict(zip(
            valid_symbols_df.get_column('symbol').to_list(),
            zip(
                valid_symbols_df.get_column('first_date').cast(str).to_list(),
                valid_symbols_df.get_column('last_date').cast(str).to_list()
            )
        ))
        
        valid_symbols = list(availability.keys())
        return valid_symbols, availability

    def load_crypto_returns(
        self,
        path: Optional[Union[str, Path]] = None,
        target_symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_null_pct: float = 0.01,
        **kwargs
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Load cryptocurrency return data from long format and convert to wide format.
        
        Args:
            path: Path to the crypto data file. If None, uses default CRYPTO_2024_LONG_PATH
            target_symbol: Symbol to use as target (e.g., 'ETH'). Will be separated from constituents.
            start_date: Start date for data range (format: 'YYYY-MM-DD')
            end_date: End date for data range (format: 'YYYY-MM-DD')
            max_null_pct: Maximum percentage of null values allowed for a symbol (0.01 = 1%)
            **kwargs: Additional arguments passed to load()
            
        Returns:
            Tuple of (target_returns, constituent_returns) as LazyFrames
            
        Raises:
            ValueError: If target_symbol is not found in the data
            ValueError: If target_symbol has too many nulls in the date range
        """
        path = path or DataPaths.CRYPTO_2024_LONG_PATH
        
        # Load long-format data
        df = pl.scan_parquet(path).collect()
        
        # Get valid symbols and their availability
        valid_symbols, availability = self.get_crypto_symbol_availability(
            df, 
            start_date=start_date, 
            end_date=end_date,
            max_null_pct=max_null_pct
        )
        
        if target_symbol:
            if target_symbol not in availability:
                raise ValueError(
                    f"Target symbol '{target_symbol}' not found in data or has too many null values "
                    f"(max allowed: {max_null_pct:.1%})"
                )
        
        # Filter by date range and valid symbols
        df_filtered = df.filter(pl.col('symbol').is_in(valid_symbols))
        if start_date:
            df_filtered = df_filtered.filter(pl.col('timestamp').cast(pl.Date) >= pl.lit(start_date).cast(pl.Date))
        if end_date:
            df_filtered = df_filtered.filter(pl.col('timestamp').cast(pl.Date) <= pl.lit(end_date).cast(pl.Date))
        
        # Process data
        df_filtered = (
            df_filtered
            .drop('price')
            .select(['timestamp', 'symbol', 'pct_return'])
        )
        
        # Pivot to wide format
        wide_df = df_filtered.pivot(
            values='pct_return',
            index='timestamp',
            columns='symbol'
        ).sort('timestamp')
        
        # Forward fill nulls within each symbol's column
        wide_df = wide_df.sort('timestamp').fill_null(0.0)
        
        # Convert back to LazyFrame
        wide_lf = wide_df.lazy()
        
        if target_symbol:
            # Split into target and constituents
            target_lf = wide_lf.select(['timestamp', target_symbol])
            constituents_lf = wide_lf.select(
                ['timestamp'] + 
                [col for col in wide_df.columns 
                 if col not in ['timestamp', target_symbol]]
            )
            return target_lf, constituents_lf
        
        return wide_lf.select('timestamp'), wide_lf.drop('timestamp')

    def load_sp500_2010_2015(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        date_col: str = 'Date',
        **kwargs
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Load 2010-2015 SP500 return and feature data.
        
        Args:
            start_date: Start date for data range (format: 'YYYY-MM-DD')
            end_date: End date for data range (format: 'YYYY-MM-DD')
            **kwargs: Additional arguments passed to load()
            
        Returns:
            Tuple of (returns_lf, features_lf) as LazyFrames
        """
        # Load returns and features
        returns_lf = pl.scan_parquet(DataPaths.SP500_RETURNS_PATH)
        features_lf = pl.scan_parquet(DataPaths.SP500_COMPONENTS_PATH)
        
        # Apply date filters if specified
        if start_date or end_date:
            date_filter = []
            if start_date:
                date_filter.append(pl.col(date_col).cast(pl.Date) >= pl.lit(start_date).cast(pl.Date))
            if end_date:
                date_filter.append(pl.col(date_col).cast(pl.Date) <= pl.lit(end_date).cast(pl.Date))
            
            returns_lf = returns_lf.filter(pl.all_horizontal(date_filter))
            features_lf = features_lf.filter(pl.all_horizontal(date_filter))
        
        return returns_lf, features_lf

    def load_data(
        self,
        data_type: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
        target_symbol: Optional[str] = None,
        **kwargs
    ) -> Union[Tuple[pl.LazyFrame, Optional[str]], Tuple[pl.LazyFrame, pl.LazyFrame]]:
        """
        Load financial data from standard paths or custom path.
        """
        if path is not None:
            return self.load(path, **kwargs)
        
        if data_type is None:
            raise ValueError("Either data_type or path must be provided")
            
        # Handle special data types
        if data_type == self.CRYPTO_RETURNS:
            return self.load_crypto_returns(target_symbol=target_symbol, **kwargs)
        elif data_type == self.SP500_2010_2015:
            return self.load_sp500_2010_2015(**kwargs)
            
        # Map data types to paths
        path_map = {
            self.SP500_INDEX: DataPaths.INDEX_2010_SP500_PATH,
            self.SP500_CONSTITUENTS: DataPaths.INDEX_2010_X_PATH
        }
        
        if data_type not in path_map:
            raise ValueError(
                f"Unknown data type: {data_type}. "
                f"Available types: {list(path_map.keys()) + [self.CRYPTO_RETURNS, self.SP500_2010_2015]}"
            )
        
        return self.load(path_map[data_type], **kwargs)