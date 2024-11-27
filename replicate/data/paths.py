from pathlib import Path
from typing import Union, Optional
import os

class DataPaths:
    """Manages data paths for the project."""
    
    # Get the path to the project root (one level up from the replicate package)
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    
    # Standard data paths
    INDEX_2010_X_PATH = DATA_DIR / "INDEX_2010_X.parquet"
    INDEX_2010_SP500_PATH = DATA_DIR / "INDEX_2010_SP500.parquet"
    CRYPTO_2024_LONG_PATH = DATA_DIR / "CRYPTO_2024_LONG.parquet"
    
    @classmethod
    def get_data_path(cls, filename: str) -> Path:
        """
        Get the full path for a data file.
        
        Args:
            filename: Name of the data file
            
        Returns:
            Full path to the data file
            
        Raises:
            FileNotFoundError: If the data directory doesn't exist
        """
        if not cls.DATA_DIR.exists():
            raise FileNotFoundError(f"Data directory not found: {cls.DATA_DIR}")
        return cls.DATA_DIR / filename
    
    @classmethod
    def ensure_data_dir(cls) -> None:
        """Ensure the data directory exists."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def list_data_files(cls, pattern: Optional[str] = "*.parquet") -> list[Path]:
        """
        List all data files matching the pattern.
        
        Args:
            pattern: Glob pattern to match files (default: "*.parquet")
            
        Returns:
            List of paths to matching data files
        """
        return list(cls.DATA_DIR.glob(pattern))
    
    @classmethod
    def validate_paths(cls) -> bool:
        """
        Validate that all standard data paths exist.
        
        Returns:
            True if all paths exist, False otherwise
        """
        standard_paths = [
            cls.INDEX_2010_X_PATH,
            cls.INDEX_2010_SP500_PATH
        ]
        return all(path.exists() for path in standard_paths)
