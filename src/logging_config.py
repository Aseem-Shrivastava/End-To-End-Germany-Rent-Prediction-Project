from pathlib import Path
from loguru import logger

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = PROJ_ROOT / "logs" / "project.log"
LOG_FILE.parent.mkdir(
    exist_ok=True, parents=True
)  # Create logs directory if it doesn't exist

# Configure Loguru
logger.remove()  # Remove the default logger
logger.add(
    LOG_FILE,
    rotation="1 MB",  # Rotate log file when it reaches 1 MB
    retention="7 days",  # Retain log files for 7 days
    level="INFO",  # Log messages of INFO level or higher
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

# tqdm compatibility
try:
    from tqdm import tqdm

    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
