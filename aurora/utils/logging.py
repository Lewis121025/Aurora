from __future__ import annotations

import logging
import sys
from typing import Any, Dict


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stdout,
    )


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {"event": event, **fields}
    # keep it simple; production can use structlog / json logger
    logger.info(payload)
