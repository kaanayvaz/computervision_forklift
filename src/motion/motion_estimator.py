"""
Motion estimation module for velocity calculation.

Computes velocity from bounding box displacement across frames with
temporal smoothing to reduce noise.
"""

import numpy as np
from typing import Optional
from collections import deque

from core.entities import TrackedObject, BoundingBox
from core.utils import get_logger

logger = get_logger(__name__)


class MotionEstimator:
    """
    Estimate motion/velocity from tracked object bbox displacement.
    
    Computes velocity as pixel displacement per frame and applies
    temporal smoothing to reduce noise from detection jitter.
    
    Args:
        smoothing_window: Number of frames for moving average smoothing.
        min_history_length: Minimum detections needed to estimate velocity.
        
    Example:
        >>> estimator = MotionEstimator(smoothing_window=5)
        >>> velocity = estimator.compute_velocity(tracked_object)
    """
    
    def __init__(
        self,
        smoothing_window: int = 5,
        min_history_length: int = 3
    ):
        self.smoothing_window = smoothing_window
        self.min_history_length = min_history_length
        
        # Per-track velocity history for smoothing
        self._velocity_history: dict[int, deque] = {}
        
        logger.info(
            f"MotionEstimator initialized: "
            f"window={smoothing_window}, min_history={min_history_length}"
        )
    
    def compute_velocity(
        self,
        track: TrackedObject
    ) -> float:
        """
        Compute current velocity for a tracked object.
        
        Args:
            track: TrackedObject with detection history.
            
        Returns:
            Velocity in pixels per frame. Returns 0 if insufficient history.
        """
        if track.detection_count < self.min_history_length:
            return 0.0
        
        # Get last two detections for instantaneous velocity
        detections = track.detections
        if len(detections) < 2:
            return 0.0
        
        det_curr = detections[-1]
        det_prev = detections[-2]
        
        # Compute center displacement
        curr_center = det_curr.bbox.center
        prev_center = det_prev.bbox.center
        
        # Frame difference (in case of skipped frames)
        frame_diff = det_curr.frame_id - det_prev.frame_id
        if frame_diff <= 0:
            frame_diff = 1
        
        # Euclidean displacement
        displacement = (
            (curr_center[0] - prev_center[0]) ** 2 +
            (curr_center[1] - prev_center[1]) ** 2
        ) ** 0.5
        
        # Velocity in pixels per frame
        instantaneous_velocity = displacement / frame_diff
        
        # Apply smoothing
        smoothed_velocity = self._smooth_velocity(
            track.track_id, instantaneous_velocity
        )
        
        return smoothed_velocity
    
    def _smooth_velocity(
        self,
        track_id: int,
        velocity: float
    ) -> float:
        """Apply moving average smoothing to velocity."""
        if track_id not in self._velocity_history:
            self._velocity_history[track_id] = deque(maxlen=self.smoothing_window)
        
        history = self._velocity_history[track_id]
        history.append(velocity)
        
        # Return moving average
        return sum(history) / len(history)
    
    def is_stationary(
        self,
        track: TrackedObject,
        idle_threshold: float = 2.0
    ) -> bool:
        """
        Check if tracked object is stationary.
        
        Args:
            track: TrackedObject to check.
            idle_threshold: Velocity threshold (pixels/frame) below which is idle.
            
        Returns:
            True if velocity is below threshold.
        """
        velocity = self.compute_velocity(track)
        return velocity < idle_threshold
    
    def compute_direction(
        self,
        track: TrackedObject
    ) -> Optional[tuple[float, float]]:
        """
        Compute movement direction vector.
        
        Args:
            track: TrackedObject with detection history.
            
        Returns:
            Normalized direction vector (dx, dy), or None if stationary.
        """
        if track.detection_count < 2:
            return None
        
        detections = track.detections
        det_curr = detections[-1]
        det_prev = detections[-2]
        
        curr_center = det_curr.bbox.center
        prev_center = det_prev.bbox.center
        
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        
        # Normalize
        magnitude = (dx ** 2 + dy ** 2) ** 0.5
        if magnitude < 1.0:  # Effectively stationary
            return None
        
        return (dx / magnitude, dy / magnitude)
    
    def get_velocity_history(
        self,
        track_id: int
    ) -> list[float]:
        """Get velocity history for a track."""
        if track_id in self._velocity_history:
            return list(self._velocity_history[track_id])
        return []
    
    def clear_track(self, track_id: int) -> None:
        """Clear velocity history for a track."""
        if track_id in self._velocity_history:
            del self._velocity_history[track_id]
    
    def reset(self) -> None:
        """Reset all velocity history."""
        self._velocity_history.clear()
