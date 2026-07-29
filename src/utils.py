import logging
import sys
from pathlib import Path


def get_logger(name=__name__):
    """Basic console logger, replaces scattered print() calls."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path):
    """Create a directory if it doesn't exist, return Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_intermediate_files(*paths):
    """Delete intermediate audio files (raw download, wav, normalized) once pipeline is done."""
    for p in paths:
        p = Path(p)
        if p.exists():
            p.unlink()


def read_text_file(path):
    return Path(path).read_text(encoding="utf-8")


def write_text_file(path, content):
    Path(path).write_text(content, encoding="utf-8")
