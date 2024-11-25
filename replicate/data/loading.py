import polars as pl
from typing import Optional, Tuple

def load(
    path: str,
    date_col: str = 'Date',
    parse_dates: bool = True,
    return_col: Optional[str] = None
) -> Tuple[pl.LazyFrame, Optional[str]]:
    """
    Lazily load return data from parquet file.
    
    Args:
        path: Path to parquet file
        date_col: Name of date column
        parse_dates: Whether to parse dates from strings
        return_col: If specified, only this return column will be kept
    
    Returns:
        LazyFrame with date column first, followed by all return columns
            (or just the specified return column if return_col is provided)
        Name of return column used (only if return_col was specified)
    """
    lf = pl.scan_parquet(path)
    
    if return_col is not None:
        if return_col not in lf.columns:
            raise ValueError(f"Column {return_col} not found in data")
        lf = lf.select([date_col, return_col])
    else:
        cols = [date_col] + [c for c in lf.collect_schema().names() if c != date_col]
        lf = lf.select(cols)
    
    if parse_dates:
        lf = lf.with_columns(pl.col(date_col).str.strptime(pl.Date, format=None))
    
    return lf, return_col