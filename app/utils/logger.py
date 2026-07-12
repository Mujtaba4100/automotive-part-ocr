"""
Centralised logging configuration.

Usage::

    from app.utils.logger import get_logger
    log = get_logger(__name__)
"""

import logging
import sys
from app.utils.constants import LOG_FORMAT, LOG_DATE_FORMAT


def get_logger(name: str = __name__) -> logging.Logger:
    """Return a named logger configured with a stream handler.

    Calling this multiple times with the same *name* always returns the
    same :class:`logging.Logger` instance (Python's logging module is
    idempotent by design).

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
