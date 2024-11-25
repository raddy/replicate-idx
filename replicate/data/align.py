import polars as pl
from typing import Tuple, Optional, List
from datetime import datetime

def align(
    lf1: pl.LazyFrame,
    lf2: pl.LazyFrame,
    on: str,
    sort: bool = True,
    validate: bool = True
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Align two lazy frames on a common column.
    
    Args:
        lf1: First LazyFrame
        lf2: Second LazyFrame
        on: Column to align on (typically a date column)
        sort: Whether to sort by the alignment column
        validate: If True, performs validation checks on the alignment
        
    Returns:
        Tuple of aligned LazyFrames
        
    Raises:
        ValueError: If validation fails or if input frames are empty
    """
    # Input validation
    if validate:
        for i, lf in enumerate([lf1, lf2], 1):
            columns = lf.collect_schema().names()
            if lf.collect().height == 0:
                raise ValueError(f"LazyFrame {i} is empty")
            if on not in columns:
                raise ValueError(f"Column '{on}' not found in LazyFrame {i}")
    
    # Find common values
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
    
    # Validate alignment results
    if validate and len(common_vals) == 0:
        raise ValueError("No common values found for alignment")
    
    # Align the frames
    lf1_aligned = lf1.filter(pl.col(on).is_in(common_vals))
    lf2_aligned = lf2.filter(pl.col(on).is_in(common_vals))
    
    if validate:
        # Ensure both frames have the same length and are properly aligned
        df1, df2 = lf1_aligned.collect(), lf2_aligned.collect()
        if df1.height != df2.height:
            raise ValueError("Aligned frames have different lengths")
        if not (df1[on] == df2[on]).all():
            raise ValueError("Frames are not properly aligned")
    
    return lf1_aligned, lf2_aligned

def align_multiple(
    frames: List[pl.LazyFrame],
    on: str,
    sort: bool = True,
    validate: bool = True
) -> List[pl.LazyFrame]:
    """
    Align multiple lazy frames on a common column.
    
    Args:
        frames: List of LazyFrames to align
        on: Column to align on
        sort: Whether to sort by the alignment column
        validate: If True, performs validation checks
        
    Returns:
        List of aligned LazyFrames
    """
    if len(frames) < 2:
        return frames
    
    result = frames[0]
    for frame in frames[1:]:
        result, frame = align(result, frame, on, sort, validate)
        frames[frames.index(frame)] = frame
    
    return frames