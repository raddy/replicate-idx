import polars as pl
from typing import Tuple

def align(
    lf1: pl.LazyFrame,
    lf2: pl.LazyFrame,
    on: str,
    sort: bool = True
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Align two lazy frames on a common column.
    
    Args:
        lf1: First LazyFrame
        lf2: Second LazyFrame
        on: Column to align on
        sort: Whether to sort by the alignment column
        
    Returns:
        Tuple of aligned LazyFrames
    """
    common = (
        lf1.select(pl.col(on))
        .join(
            lf2.select(pl.col(on)),
            on=on,
            how='inner'
        )
    )
    
    if sort:
        common = common.sort(on)
    common_vals = common.collect()[on]

    lf1 = lf1.filter(pl.col(on).is_in(common_vals))
    lf2 = lf2.filter(pl.col(on).is_in(common_vals))
    return lf1, lf2