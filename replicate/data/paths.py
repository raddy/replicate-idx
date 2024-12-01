from pathlib import Path
from typing import Union, Optional, List

class DataPaths:
    """Manages data paths for the project."""
    
    # Get the path to the project root (one level up from the replicate package)
    ROOT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = ROOT_DIR / "data"
    
    CRYPTO_2024_LONG_PATH = DATA_DIR / "CRYPTO_2024_LONG.parquet"
    
    # SP500 data paths
    SP500_2010_X_PATH = DATA_DIR / "SP500_X_2010.parquet"
    SP500_2010_R_PATH = DATA_DIR / "SP500_r_2010.parquet"
    SP500_2010_2015_X_PATH = DATA_DIR / "SP500_r_2010_2015.parquet"
    SP500_2010_2015_R_PATH = DATA_DIR / "SP500_X_2010_2015.parquet"
    
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
    def get_data_files(cls, pattern: str) -> List[Path]:
        """
        Get all data files matching the given pattern.
        
        Args:
            pattern: Glob pattern to match files
            
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
            cls.SP500_2010_X_PATH,
            cls.SP500_2010_R_PATH,
            cls.SP500_2010_2015_X_PATH,
            cls.SP500_2010_2015_R_PATH,
            cls.CRYPTO_2024_LONG_PATH
        ]
        return all(path.exists() for path in standard_paths)
