import polars as pl
from pathlib import Path
from typing import Optional, Tuple, Union, Literal
from .paths import DataPaths

class DataLoader:
    """Data loading utilities for financial time series."""
    
    # Standard data types
    SP500_INDEX = "SP500_INDEX"
    SP500_CONSTITUENTS = "SP500_CONSTITUENTS"
    
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
    
    def load_data(
        self,
        data_type: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Tuple[pl.LazyFrame, Optional[str]]:
        """
        Load financial data from standard paths or custom path.
        
        Args:
            data_type: Type of data to load (e.g., "SP500_INDEX", "SP500_CONSTITUENTS")
                      If None, path must be provided
            path: Custom path to load data from
                 If provided, data_type is ignored
            **kwargs: Additional arguments passed to load()
            
        Returns:
            LazyFrame with loaded data and optional return column name
            
        Raises:
            ValueError: If neither data_type nor path is provided
        """
        if path is not None:
            return self.load(path, **kwargs)
        
        if data_type is None:
            raise ValueError("Either data_type or path must be provided")
            
        # Map data types to paths
        path_map = {
            self.SP500_INDEX: DataPaths.INDEX_2010_SP500_PATH,
            self.SP500_CONSTITUENTS: DataPaths.INDEX_2010_X_PATH
        }
        
        if data_type not in path_map:
            raise ValueError(
                f"Unknown data type: {data_type}. "
                f"Available types: {list(path_map.keys())}"
            )
        
        return self.load(path_map[data_type], **kwargs)