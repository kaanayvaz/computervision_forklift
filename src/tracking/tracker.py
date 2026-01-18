"""
Object tracking module using ByteTrack via supervision library.

Provides ForkliftTracker class that:
- Takes detections from ForkliftDetector
- Assigns persistent track IDs
- Maintains track history
- Handles track lifecycle (creation, update, termination)
"""

import numpy as np
from typing import Optional
from collections import defaultdict

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    sv = None

from core.entities import Detection, TrackedObject, BoundingBox, ForkliftState
from core.utils import get_logger

logger = get_logger(__name__)


class ForkliftTracker:
    """
    Multi-object tracker for forklifts using ByteTrack algorithm.
    
    Assigns persistent IDs to detected objects across frames and maintains
    track history for velocity estimation and state classification.
    
    Args:
        track_activation_threshold: Confidence threshold to create new tracks.
        lost_track_buffer: Frames to wait before deleting lost track.
        minimum_matching_threshold: IoU threshold for matching detections to tracks.
        frame_rate: Video frame rate for ByteTrack internals.
        
    Example:
        >>> tracker = ForkliftTracker()
        >>> for frame_id, detections in enumerate(all_detections):
        ...     tracked_objects = tracker.update(detections, frame_id)
        ...     for obj in tracked_objects:
        ...         print(f"Track {obj.track_id}: {obj.latest_bbox}")
    """
    
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30
    ):
        if not SUPERVISION_AVAILABLE:
            raise ImportError(
                "supervision library is required for tracking. "
                "Install with: pip install supervision"
            )
        
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.frame_rate = frame_rate
        
        # Initialize ByteTrack
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate
        )
        
        # Track storage: track_id -> TrackedObject
        self._tracks: dict[int, TrackedObject] = {}
        
        # Active track IDs in current frame
        self._active_track_ids: set[int] = set()
        
        # Current frame ID
        self._current_frame_id: int = 0
        
        logger.info(
            f"Tracker initialized: "
            f"threshold={track_activation_threshold}, "
            f"buffer={lost_track_buffer}, "
            f"matching={minimum_matching_threshold}"
        )
    
    def update(
        self,
        detections: list[Detection],
        frame_id: int
    ) -> list[TrackedObject]:
        """
        Update tracker with new detections and return tracked objects.
        
        Args:
            detections: List of Detection objects from current frame.
            frame_id: Current frame index.
            
        Returns:
            List of TrackedObject with updated track IDs.
        """
        self._current_frame_id = frame_id
        
        if not detections:
            self._active_track_ids = set()
            return []
        
        # Convert detections to supervision format
        sv_detections = self._detections_to_supervision(detections)
        
        # Run ByteTrack
        tracked_detections = self._tracker.update_with_detections(sv_detections)
        
        # Convert results back to TrackedObject
        tracked_objects = self._process_tracked_detections(
            tracked_detections, detections, frame_id
        )
        
        logger.debug(
            f"Frame {frame_id}: {len(detections)} detections -> "
            f"{len(tracked_objects)} tracks"
        )
        
        return tracked_objects
    
    def _detections_to_supervision(
        self,
        detections: list[Detection]
    ) -> 'sv.Detections':
        """Convert Detection list to supervision Detections format."""
        if not detections:
            return sv.Detections.empty()
        
        xyxy = np.array([
            [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]
            for d in detections
        ])
        
        confidence = np.array([d.confidence for d in detections])
        class_ids = np.array([d.class_id for d in detections])
        
        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids
        )
    
    def _process_tracked_detections(
        self,
        tracked_detections: 'sv.Detections',
        original_detections: list[Detection],
        frame_id: int
    ) -> list[TrackedObject]:
        """Process ByteTrack results and update TrackedObject storage."""
        tracked_objects = []
        new_active_ids = set()
        
        if tracked_detections.tracker_id is None:
            return tracked_objects
        
        for i, track_id in enumerate(tracked_detections.tracker_id):
            track_id = int(track_id)
            new_active_ids.add(track_id)
            
            # Get detection info
            xyxy = tracked_detections.xyxy[i]
            confidence = tracked_detections.confidence[i] if tracked_detections.confidence is not None else 0.5
            class_id = tracked_detections.class_id[i] if tracked_detections.class_id is not None else 0
            
            # Find matching original detection for class name
            class_name = self._find_class_name(xyxy, original_detections)
            
            # Create detection object
            detection = Detection(
                bbox=BoundingBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3])
                ),
                class_id=int(class_id),
                class_name=class_name,
                confidence=float(confidence),
                frame_id=frame_id
            )
            
            # Get or create TrackedObject
            if track_id not in self._tracks:
                self._tracks[track_id] = TrackedObject(
                    track_id=track_id,
                    class_name=class_name
                )
                logger.debug(f"New track created: ID={track_id}, class={class_name}")
            
            tracked_obj = self._tracks[track_id]
            tracked_obj.add_detection(detection)
            tracked_objects.append(tracked_obj)
        
        self._active_track_ids = new_active_ids
        
        return tracked_objects
    
    def _find_class_name(
        self,
        xyxy: np.ndarray,
        detections: list[Detection]
    ) -> str:
        """Find matching detection and return its class name."""
        # Find detection with closest bbox
        min_dist = float('inf')
        best_class = "unknown"
        
        for det in detections:
            dist = (
                (det.bbox.x1 - xyxy[0]) ** 2 +
                (det.bbox.y1 - xyxy[1]) ** 2 +
                (det.bbox.x2 - xyxy[2]) ** 2 +
                (det.bbox.y2 - xyxy[3]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                best_class = det.class_name
        
        return best_class
    
    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """
        Get TrackedObject by track ID.
        
        Args:
            track_id: Track identifier.
            
        Returns:
            TrackedObject if found, None otherwise.
        """
        return self._tracks.get(track_id)
    
    def get_active_tracks(self) -> list[TrackedObject]:
        """
        Get all currently active tracks.
        
        Returns:
            List of TrackedObject that were updated in the last frame.
        """
        return [
            self._tracks[tid]
            for tid in self._active_track_ids
            if tid in self._tracks
        ]
    
    def get_all_tracks(self) -> list[TrackedObject]:
        """
        Get all tracks (including inactive).
        
        Returns:
            List of all TrackedObject instances.
        """
        return list(self._tracks.values())
    
    def get_forklift_tracks(self) -> list[TrackedObject]:
        """Get only forklift tracks."""
        return [t for t in self.get_active_tracks() if t.class_name == "forklift"]
    
    def reset(self) -> None:
        """Reset tracker state."""
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=self.lost_track_buffer,
            minimum_matching_threshold=self.minimum_matching_threshold,
            frame_rate=self.frame_rate
        )
        self._tracks.clear()
        self._active_track_ids.clear()
        self._current_frame_id = 0
        logger.info("Tracker reset")
    
    @property
    def track_count(self) -> int:
        """Total number of tracks created."""
        return len(self._tracks)
    
    @property
    def active_track_count(self) -> int:
        """Number of currently active tracks."""
        return len(self._active_track_ids)
