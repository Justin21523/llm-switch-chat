"""Performance monitoring utilities."""

# backend/app/utils/timing.py
import time
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise

    return wrapper


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Timer", log_level: int = logging.INFO):
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time

        if exc_type is not None:
            logger.log(self.log_level, f"{self.name} failed after {elapsed:.3f}s")
        else:
            logger.log(self.log_level, f"{self.name} completed in {elapsed:.3f}s")

    @property
    def elapsed(self) -> float:
        """Get elapsed time."""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
