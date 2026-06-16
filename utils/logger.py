"""utils/logger.py – centralised logging."""
import logging, sys, os

_LEVEL = os.environ.get("AMR_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
