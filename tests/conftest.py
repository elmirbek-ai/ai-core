import logging
import os

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("AI_CORE_AUTH_ENABLED", "false")
os.environ.setdefault("AI_CORE_RATE_LIMIT_ENABLED", "false")


@pytest.fixture(autouse=True)
def capture_application_logs(caplog):
    # Application JSON output does not propagate to root (prevents duplicate
    # production lines). Attach pytest's capture handler at that boundary.
    logger = logging.getLogger("app")
    caplog.set_level(logging.INFO, logger="app")
    previous_propagation = logger.propagate
    logger.propagate = False
    logger.addHandler(caplog.handler)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)
        logger.propagate = previous_propagation
