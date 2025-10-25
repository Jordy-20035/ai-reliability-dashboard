"""
Logging configuration for the monitoring system.
"""

import sys
from pathlib import Path
from loguru import logger

# Remove default handler
logger.remove()

# Create logs directory
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Console handler with custom format
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# File handler for all logs
logger.add(
    log_dir / "app.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
)

# File handler for errors only
logger.add(
    log_dir / "errors.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
    rotation="10 MB",
    retention="90 days",
    compression="zip",
)


def get_logger(name: str = __name__):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        logger: Configured logger instance
    """
    return logger.bind(name=name)


# Export logger
__all__ = ["logger", "get_logger"]

