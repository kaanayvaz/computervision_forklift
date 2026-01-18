"""
Activity segmentation module.

Groups frames with same state into temporal activity segments.
Identifies non-value-added activities based on rules.
"""

from dataclasses import dataclass
from typing import Optional

from core.entities import Activity, TrackedObject, ForkliftState
from core.utils import get_logger

logger = get_logger(__name__)


class ActivitySegmenter:
    """
    Segment continuous state periods into discrete activities.
    
    Groups consecutive frames with the same state into Activity objects.
    Merges short segments and identifies non-value-added activities.
    
    Args:
        min_duration: Minimum activity duration (seconds) to include.
        merge_threshold: Merge activities shorter than this (seconds).
        fps: Video frame rate for duration calculations.
    """
    
    def __init__(
        self,
        min_duration: float = 5.0,
        merge_threshold: float = 3.0,
        fps: float = 30.0
    ):
        self.min_duration = min_duration
        self.merge_threshold = merge_threshold
        self.fps = fps
        
        # Non-value-added thresholds
        self.idle_threshold_seconds = 30.0
        self.blocked_threshold_seconds = 15.0
        self.absent_threshold_seconds = 60.0
        
        logger.info(
            f"ActivitySegmenter initialized: "
            f"min_duration={min_duration}s, merge_threshold={merge_threshold}s"
        )
    
    def segment(
        self,
        track: TrackedObject,
        fps: Optional[float] = None
    ) -> list[Activity]:
        """
        Segment a track's state history into activities.
        
        Args:
            track: TrackedObject with state history.
            fps: Optional FPS override.
            
        Returns:
            List of Activity objects representing state periods.
        """
        if fps is None:
            fps = self.fps
        
        if not track.state_history:
            return []
        
        activities = []
        current_state = None
        start_frame = 0
        
        # Process state history
        for i, state in enumerate(track.state_history):
            if current_state is None:
                current_state = state
                start_frame = i
            elif state != current_state:
                # State changed - create activity for previous segment
                activity = self._create_activity(
                    track.track_id,
                    current_state,
                    start_frame,
                    i - 1,
                    fps
                )
                activities.append(activity)
                
                # Start new segment
                current_state = state
                start_frame = i
        
        # Handle final segment
        if current_state is not None:
            activity = self._create_activity(
                track.track_id,
                current_state,
                start_frame,
                len(track.state_history) - 1,
                fps
            )
            activities.append(activity)
        
        # Merge short segments
        activities = self._merge_short_segments(activities, fps)
        
        # Filter by minimum duration
        activities = [a for a in activities if a.duration_seconds >= self.min_duration]
        
        # Classify value-added status
        for activity in activities:
            activity.is_value_added = self._is_value_added(activity)
        
        return activities
    
    def _create_activity(
        self,
        track_id: int,
        state: ForkliftState,
        start_frame: int,
        end_frame: int,
        fps: float
    ) -> Activity:
        """Create Activity object from frame range."""
        return Activity(
            track_id=track_id,
            state=state,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            is_value_added=True  # Will be set later
        )
    
    def _merge_short_segments(
        self,
        activities: list[Activity],
        fps: float
    ) -> list[Activity]:
        """Merge very short segments with neighbors."""
        if len(activities) <= 1:
            return activities
        
        merged = []
        i = 0
        
        while i < len(activities):
            current = activities[i]
            
            if current.duration_seconds < self.merge_threshold and i < len(activities) - 1:
                # Merge with next activity if they have same state
                next_act = activities[i + 1]
                if current.state == next_act.state:
                    merged_activity = Activity(
                        track_id=current.track_id,
                        state=current.state,
                        start_frame=current.start_frame,
                        end_frame=next_act.end_frame,
                        fps=fps
                    )
                    merged.append(merged_activity)
                    i += 2
                    continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def _is_value_added(self, activity: Activity) -> bool:
        """
        Determine if activity is value-added.
        
        Non-value-added activities:
        - IDLE for extended period
        - UNKNOWN state
        """
        if activity.state == ForkliftState.IDLE:
            # Long idle periods are non-value-added
            if activity.duration_seconds >= self.idle_threshold_seconds:
                return False
        
        if activity.state == ForkliftState.UNKNOWN:
            return False
        
        # Moving and loading/unloading are value-added
        return activity.state in [
            ForkliftState.MOVING_EMPTY,
            ForkliftState.MOVING_LOADED,
            ForkliftState.LOADING,
            ForkliftState.UNLOADING
        ]
    
    def segment_all_tracks(
        self,
        tracks: list[TrackedObject],
        fps: Optional[float] = None
    ) -> dict[int, list[Activity]]:
        """
        Segment activities for all tracks.
        
        Args:
            tracks: List of TrackedObjects.
            fps: Optional FPS override.
            
        Returns:
            Dictionary mapping track_id to list of activities.
        """
        results = {}
        
        for track in tracks:
            activities = self.segment(track, fps)
            results[track.track_id] = activities
        
        return results
    
    def get_idle_activities(
        self,
        activities: list[Activity]
    ) -> list[Activity]:
        """Get only idle activities."""
        return [a for a in activities if a.state == ForkliftState.IDLE]
    
    def get_non_value_added(
        self,
        activities: list[Activity]
    ) -> list[Activity]:
        """Get non-value-added activities."""
        return [a for a in activities if not a.is_value_added]
    
    def total_time_by_state(
        self,
        activities: list[Activity]
    ) -> dict[ForkliftState, float]:
        """
        Calculate total time spent in each state.
        
        Returns:
            Dictionary mapping state to total seconds.
        """
        totals: dict[ForkliftState, float] = {}
        
        for activity in activities:
            if activity.state not in totals:
                totals[activity.state] = 0.0
            totals[activity.state] += activity.duration_seconds
        
        return totals
