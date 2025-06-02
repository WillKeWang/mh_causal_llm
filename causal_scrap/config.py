import os
from pathlib import Path

# Environment variable overrides, else fall back to repo‑relative path
GLOBEM_DATA_DIR = Path(
    os.getenv("GLOBEM_DATA_DIR", Path(__file__).resolve().parent / "data/globem_raw")
)