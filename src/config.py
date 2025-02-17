from pathlib import Path

from dotenv import load_dotenv
from src.logging_config import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

ARTIFACTS_DIR = PROJ_ROOT / "artifacts"
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTRACTED_DATA_DIR = DATA_DIR / "extracted"

MODELS_DIR = PROJ_ROOT / "models"
