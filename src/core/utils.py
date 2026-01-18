"""
Utility functions for forklift analytics system.

Provides:
- Configuration loading
- Logging setup
- Geometry helpers
- Timing utilities
"""

import logging
import time
import yaml
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Dictionary with configuration values.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def merge_configs(base: dict, override: dict) -> dict:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base: Base configuration.
        override: Override configuration (takes precedence).
        
    Returns:
        Merged configuration dictionary.
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_nested_config(config: dict, path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary.
        path: Dot-separated path (e.g., "model.confidence_threshold").
        default: Default value if path not found.
        
    Returns:
        Configuration value or default.
        
    Example:
        >>> config = {"model": {"threshold": 0.5}}
        >>> get_nested_config(config, "model.threshold", 0.3)
        0.5
    """
    keys = path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.
        format_string: Optional custom format string.
        
    Returns:
        Configured logger instance.
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create handlers
    handlers: list[logging.Handler] = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True
    )
    
    # Return logger for forklift_analytics
    logger = logging.getLogger("forklift_analytics")
    logger.setLevel(numeric_level)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name (usually __name__).
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(f"forklift_analytics.{name}")


# ============================================================================
# Geometry Helpers
# ============================================================================

def compute_iou(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float]
) -> float:
    """
    Compute Intersection over Union between two bounding boxes.
    
    Args:
        box1: First box as (x1, y1, x2, y2).
        box2: Second box as (x1, y1, x2, y2).
        
    Returns:
        IoU value between 0 and 1.
    """
    # Compute intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check for no intersection
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Compute areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute union
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_containment(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float]
) -> float:
    """
    Compute what fraction of inner box is contained within outer box.
    
    Args:
        inner: Inner box as (x1, y1, x2, y2).
        outer: Outer box as (x1, y1, x2, y2).
        
    Returns:
        Containment ratio between 0 and 1.
    """
    # Compute intersection
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    
    # Check for no intersection
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    
    if inner_area == 0:
        return 0.0
    
    return intersection / inner_area


def compute_distance(
    point1: tuple[float, float],
    point2: tuple[float, float]
) -> float:
    """
    Compute Euclidean distance between two points.
    
    Args:
        point1: First point as (x, y).
        point2: Second point as (x, y).
        
    Returns:
        Euclidean distance.
    """
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def point_in_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]]
) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: Point as (x, y).
        polygon: List of polygon vertices as [(x, y), ...].
        
    Returns:
        True if point is inside polygon.
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        
        j = i
    
    return inside


# ============================================================================
# Timing Utilities
# ============================================================================

class Timer:
    """Simple timing utility for performance measurement."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        
        self.elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return self.elapsed
    
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000


@contextmanager
def timed_block(name: str, logger: Optional[logging.Logger] = None):
    """
    Context manager for timing a block of code.
    
    Args:
        name: Name of the operation for logging.
        logger: Optional logger instance.
        
    Yields:
        Timer instance.
        
    Example:
        >>> with timed_block("Processing frame") as timer:
        ...     # do work
        ... print(f"Took {timer.elapsed_ms():.2f}ms")
    """
    timer = Timer()
    timer.start()
    
    try:
        yield timer
    finally:
        elapsed = timer.stop()
        
        if logger:
            logger.debug(f"{name}: {timer.elapsed_ms():.2f}ms")


class FPSTracker:
    """Track frames per second over a rolling window."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times: list[float] = []
        self.last_time: Optional[float] = None
    
    def tick(self) -> None:
        """Record a frame timestamp."""
        current_time = time.perf_counter()
        
        if self.last_time is not None:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)
            
            # Keep window bounded
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
        
        self.last_time = current_time
    
    @property
    def fps(self) -> float:
        """Get current FPS estimate."""
        if not self.frame_times:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        
        if avg_frame_time == 0:
            return 0.0
        
        return 1.0 / avg_frame_time
    
    @property
    def avg_frame_time_ms(self) -> float:
        """Get average frame processing time in milliseconds."""
        if not self.frame_times:
            return 0.0
        
        return (sum(self.frame_times) / len(self.frame_times)) * 1000
