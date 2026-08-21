from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def get_logger(name: str = "shiaqa") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
        _CONFIGURED = True
        # httpx logs every request at INFO; too noisy for a 100-request crawl.
        logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger(name)
