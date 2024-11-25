from pathlib import Path

# Get the path to the project root (one level up from the replicate package)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Define paths to the parquet files
INDEX_2010_X_PATH = PROJECT_ROOT / "data" / "INDEX_2010_X.parquet"
INDEX_2010_SP500_PATH = PROJECT_ROOT / "data" / "INDEX_2010_SP500.parquet"
