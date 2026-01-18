"""
Core entities for forklift analytics system.

This module defines the core data structures used throughout the pipeline:
- Detection: Individual object detection result
- TrackedObject: Object with persistent tracking ID
- ForkliftState: Enumeration of possible forklift states
- Activity: Time-bounded activity segment
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ForkliftState(Enum):
    """Possible operational states for a forklift."""
    
    UNKNOWN = "unknown"
    IDLE = "idle"
    MOVING_EMPTY = "moving_empty"
    MOVING_LOADED = "moving_loaded"
    LOADING = "loading"
    UNLOADING = "unloading"


class ObjectClass(Enum):
    """Detected object classes."""
    
    FORKLIFT = "forklift"
    PALLET = "pallet"
    PERSON = "person"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box in XYXY format (x1, y1, x2, y2)."""
    
    x1: float
    y1: float
    x2: float
    y2: float
    
    def __post_init__(self):
        """Validate bounding box coordinates."""
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1
    
    @property
    def width(self) -> float:
        """Width of bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Height of bounding box."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """Area of bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> tuple[float, float]:
        """Center point (x, y) of bounding box."""
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2
    
    @property
    def centroid_x(self) -> float:
        """X coordinate of center."""
        return (self.x1 + self.x2) / 2
    
    @property
    def centroid_y(self) -> float:
        """Y coordinate of center."""
        return (self.y1 + self.y2) / 2
    
    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def as_xywh(self) -> tuple[float, float, float, float]:
        """Return as (x, y, width, height) tuple."""
        return (self.x1, self.y1, self.width, self.height)


@dataclass
class Detection:
    """Single object detection result."""
    
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float
    frame_id: int
    
    def __post_init__(self):
        """Validate detection values."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.frame_id < 0:
            raise ValueError(f"Frame ID must be non-negative, got {self.frame_id}")
    
    @property
    def object_class(self) -> ObjectClass:
        """Get ObjectClass enum from class_name."""
        try:
            return ObjectClass(self.class_name.lower())
        except ValueError:
            return ObjectClass.UNKNOWN


@dataclass
class TrackedObject:
    """Object with persistent tracking ID across frames."""
    
    track_id: int
    class_name: str
    detections: list[Detection] = field(default_factory=list)
    is_carrying_pallet: bool = False
    current_state: ForkliftState = ForkliftState.UNKNOWN
    velocity: float = 0.0
    
    # State history for temporal smoothing
    state_history: list[ForkliftState] = field(default_factory=list)
    
    # Associated objects
    associated_pallet_id: Optional[int] = None
    associated_operator_id: Optional[int] = None
    
    @property
    def latest_detection(self) -> Optional[Detection]:
        """Get most recent detection."""
        return self.detections[-1] if self.detections else None
    
    @property
    def latest_bbox(self) -> Optional[BoundingBox]:
        """Get most recent bounding box."""
        return self.latest_detection.bbox if self.latest_detection else None
    
    @property
    def detection_count(self) -> int:
        """Number of detections in track history."""
        return len(self.detections)
    
    @property
    def first_frame(self) -> Optional[int]:
        """First frame ID where this object was detected."""
        return self.detections[0].frame_id if self.detections else None
    
    @property
    def last_frame(self) -> Optional[int]:
        """Last frame ID where this object was detected."""
        return self.detections[-1].frame_id if self.detections else None
    
    def add_detection(self, detection: Detection) -> None:
        """Add a new detection to this track."""
        self.detections.append(detection)
    
    def update_state(self, state: ForkliftState, max_history: int = 30) -> None:
        """Update current state and maintain history."""
        self.current_state = state
        self.state_history.append(state)
        
        # Keep history bounded
        if len(self.state_history) > max_history:
            self.state_history = self.state_history[-max_history:]


@dataclass
class Activity:
    """Time-bounded activity segment for a tracked forklift."""
    
    track_id: int
    state: ForkliftState
    start_frame: int
    end_frame: int
    fps: float = 30.0
    is_value_added: bool = True
    
    # Optional metadata
    description: str = ""
    zone: str = ""
    
    def __post_init__(self):
        """Validate activity bounds."""
        if self.start_frame > self.end_frame:
            raise ValueError(f"Start frame ({self.start_frame}) must be <= end frame ({self.end_frame})")
        if self.fps <= 0:
            raise ValueError(f"FPS must be positive, got {self.fps}")
    
    @property
    def duration_frames(self) -> int:
        """Duration in frames."""
        return self.end_frame - self.start_frame + 1
    
    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.duration_frames / self.fps
    
    @property
    def is_idle(self) -> bool:
        """Check if this is an idle activity."""
        return self.state == ForkliftState.IDLE


@dataclass
class AnalyticsResult:
    """Aggregated analytics for a video or session."""
    
    total_duration_seconds: float
    total_active_time_seconds: float
    total_idle_time_seconds: float
    
    # Counts
    forklift_count: int
    activity_count: int
    
    # Utilization
    utilization_percentage: float
    
    # Cost analysis
    cost_of_waste: float
    cost_per_idle_hour: float
    
    # Breakdowns
    activities_by_state: dict[str, int] = field(default_factory=dict)
    idle_breakdown: dict[str, float] = field(default_factory=dict)  # category -> seconds
    
    @property
    def productive_time_percentage(self) -> float:
        """Percentage of time spent productively."""
        if self.total_duration_seconds == 0:
            return 0.0
        return (self.total_active_time_seconds / self.total_duration_seconds) * 100
