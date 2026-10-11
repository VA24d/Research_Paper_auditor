"""Logging for the `auditor` package, with run and task ids on every line."""

import logging
import sys
from pathlib import Path

from auditor.infra.scope import current_scope

FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(run_id)s %(task_id)s] %(message)s"


class ScopeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        active = current_scope()
        record.run_id = active.run_id if active else "-"
        record.task_id = (active.task_id if active else None) or "-"
        return True


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure the `auditor` logger. Safe to call more than once."""
    logger = logging.getLogger("auditor")
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(FORMAT)
    console = logging.StreamHandler(sys.stderr)
    handlers: list[logging.Handler] = [console]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(ScopeFilter())
        logger.addHandler(handler)
    return logger
