import logging
import os
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


def get_logger(name: str) -> logging.Logger:
    """Creates and returns a logger with the specified name.

    Args:
        name: The name of the logger.

    Returns:
        A logger instance with the specified name.
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        # Get log level from environment or default to WARNING
        log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
        level = level_map.get(log_level, logging.WARNING)

        # Configure rich console with custom theme
        theme = Theme({
            "info": "cyan",
            "warning": "yellow",
            "error": "red",
            "critical": "red bold",
            "debug": "grey50"
        })
        console = Console(theme=theme, width=100, tab_size=4)

        # Create handler for this logger
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
            omit_repeated_times=True
        )
        handler.setFormatter(logging.Formatter("WANDBOT | %(message)s"))
        
        logger.setLevel(level)
        logger.addHandler(handler)
        logger.propagate = False  # Don't propagate to root logger
    
    return logger


logger = get_logger(__name__)