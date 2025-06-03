# src/utils/logger.py
"""Provides logging utilities for the multi-agent system.

This module offers a centralized way to configure and obtain logger instances
for different components of the system, ensuring consistent log formatting and setup.
"""

import logging
from typing import Optional

ROOT_LOGGER_NAME = 'MAS_Logger'
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_LEVEL = logging.INFO

def get_logger(name: Optional[str] = None, level: int = DEFAULT_LOG_LEVEL, log_format: str = DEFAULT_LOG_FORMAT) -> logging.Logger:
    """
    Configures and returns a logger instance.

    If 'name' is None, the root MAS_Logger is returned.
    If 'name' is provided, a child logger of MAS_Logger (e.g., MAS_Logger.child_name) is returned.
    This function ensures that handlers are not added multiple times if the logger
    for the given name has already been configured.

    Args:
        name: Optional name for a child logger. If None, the root MAS_Logger is used.
        level: The logging level to set for the logger and its handler.
        log_format: The format string for log messages.

    Returns:
        A configured logging.Logger instance.
    """
    if name:
        logger_name = f"{ROOT_LOGGER_NAME}.{name}"
    else:
        logger_name = ROOT_LOGGER_NAME

    logger = logging.getLogger(logger_name)

    # Configure the logger only if it hasn't been configured before (e.g. by checking handlers)
    # This check is crucial to prevent adding multiple handlers if get_logger is called multiple times.
    if not logger.handlers:
        logger.setLevel(level)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)  # Set the level for the handler as well

        # Create formatter and add it to the handler
        formatter = logging.Formatter(log_format)
        ch.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(ch)

        # It's often good practice in libraries to not propagate messages to the root logger,
        # allowing the application using the library to control root logging.
        # However, for an application like this, default propagation might be fine.
        # logger.propagate = False # Default is True, which is usually fine for application loggers.

    # Set or update the level for the logger and its handlers.
    # This handles cases where get_logger might be called again for an existing logger
    # but with a different desired level.
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
            
    return logger

if __name__ == '__main__':
    # Example usage demonstrating the logger setup

    # Get the root MAS_Logger
    root_logger = get_logger(level=logging.DEBUG) # Test with DEBUG level for root
    root_logger.debug("MAS_Logger (root) test: Debug message.")
    root_logger.info("MAS_Logger (root) test: Info message.")

    # Get a child logger for a coordinator component
    coordinator_logger = get_logger("coordinator", level=logging.INFO)
    coordinator_logger.info("Coordinator operational.")
    coordinator_logger.debug("This coordinator debug message should not appear if its level is INFO.")

    # Get a child logger for a specific agent
    agent_logger = get_logger("agents.special_ops", level=logging.DEBUG)
    agent_logger.debug("Special Ops Agent performing debug tasks.")
    agent_logger.info("Special Ops Agent info.")

    # Test getting the root logger again (should not add new handlers or change existing ones unless level is different)
    root_logger_again = get_logger() # Defaults to MAS_Logger, level INFO
    root_logger_again.info("MAS_Logger (root) obtained again: Info message.")
    # This debug message for root_logger_again should not appear if its effective level is INFO
    # (it was set to DEBUG initially, but the handler was added then. Subsequent calls update levels)
    root_logger_again.debug("MAS_Logger (root) obtained again: Debug message. (Visibility depends on current level)")

    # Demonstrate level setting persistence and updates
    # Initially, coordinator_logger is INFO. Let's get it again but request DEBUG.
    coordinator_logger_debug_mode = get_logger("coordinator", level=logging.DEBUG)
    coordinator_logger_debug_mode.debug("Coordinator now in debug mode: This debug message should appear.")
    coordinator_logger.debug("Coordinator (original var) also reflects new DEBUG level for its logger instance.")
