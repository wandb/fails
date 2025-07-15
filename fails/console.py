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

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    # Get log level from environment or default to INFO
    log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
    level = level_map.get(log_level, logging.INFO)  # Default to INFO if invalid level

    # Configure rich console with custom theme
    theme = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "red",
        "critical": "red bold",
        "debug": "grey50"
    })
    console = Console(theme=theme, width=100, tab_size=4)

    logging.basicConfig(
        level=level,
        format="WANDBOT | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
            omit_repeated_times=True
        )]
    )
    logger = logging.getLogger(name)
    return logger


logger = get_logger(__name__)