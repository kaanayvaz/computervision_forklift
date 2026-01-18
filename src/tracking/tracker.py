"""
Object tracking module using ByteTrack via supervision library.

Provides ForkliftTracker class that:
- Takes detections from ForkliftDetector
- Assigns persistent track IDs
- Maintains track history
- Handles track lifecycle (creation, update, termination)
- Camera Motion Compensation (CMC) to handle moving cameras
"""

import numpy as np
import cv2
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


class CameraMotionCompensator:
    """
    Estimates camera motion between frames using optical flow or feature matching.
    
    Uses sparse optical flow (Lucas-Kanade) to estimate global camera motion
    and computes an affine transformation matrix to compensate detection positions.
    """
    
    def __init__(self, max_features: int = 500, quality_level: float = 0.01):
        self.max_features = max_features
        self.quality_level = quality_level
        self.prev_gray = None
        self.prev_keypoints = None
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Parameters for feature detection
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=quality_level,
            minDistance=10,
            blockSize=7
        )
        
        logger.info(f"CameraMotionCompensator initialized: max_features={max_features}")
    
    def estimate_motion(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate camera motion from previous frame to current frame.
        
        Args:
            frame: Current frame (BGR or grayscale)
            
        Returns:
            2x3 affine transformation matrix, or None if cannot estimate
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        transform = None
        
        if self.prev_gray is not None and self.prev_keypoints is not None:
            if len(self.prev_keypoints) > 0:
                # Track features using optical flow
                curr_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_keypoints, None, **self.lk_params
                )
                
                # Select good points
                if status is not None:
                    good_prev = self.prev_keypoints[status.flatten() == 1]
                    good_curr = curr_keypoints[status.flatten() == 1]
                    
                    # Need at least 4 points for affine transform
                    if len(good_prev) >= 4:
                        # Estimate affine transformation using RANSAC
                        transform, inliers = cv2.estimateAffinePartial2D(
                            good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0
                        )
                        
                        if transform is not None:
                            # Check if transform is reasonable (not too extreme)
                            dx = transform[0, 2]  # translation x
                            dy = transform[1, 2]  # translation y
                            scale = np.sqrt(transform[0, 0]**2 + transform[0, 1]**2)
                            
                            # Reject extreme transforms (scale should be ~1, translation reasonable)
                            if abs(scale - 1.0) > 0.3 or abs(dx) > 200 or abs(dy) > 200:
                                logger.debug(f"Rejected extreme transform: scale={scale:.2f}, dx={dx:.1f}, dy={dy:.1f}")
                                transform = None
                            else:
                                logger.debug(f"Camera motion: dx={dx:.1f}, dy={dy:.1f}, scale={scale:.3f}")
        
        # Update for next frame
        self.prev_gray = gray
        
        # Detect new features for next frame
        new_keypoints = cv2.goodFeaturesToTrack(gray, **self.feature_params)
        self.prev_keypoints = new_keypoints if new_keypoints is not None else np.array([])
        
        return transform
    
    def compensate_detections(
        self, 
        detections: list, 
        transform: np.ndarray
    ) -> list:
        """
        Apply inverse camera motion to detections to stabilize tracking.
        
        This transforms detection coordinates as if the camera hadn't moved,
        making it easier for the tracker to associate detections with existing tracks.
        
        Args:
            detections: List of Detection objects
            transform: 2x3 affine transformation matrix
            
        Returns:
            List of Detection objects with compensated coordinates
        """
        if transform is None or len(detections) == 0:
            return detections
        
        # Compute inverse transform
        try:
            # Add [0, 0, 1] row to make it 3x3
            full_transform = np.vstack([transform, [0, 0, 1]])
            inv_transform = np.linalg.inv(full_transform)[:2]  # Back to 2x3
        except np.linalg.LinAlgError:
            return detections
        
        compensated = []
        for det in detections:
            # Transform bbox corners
            corners = np.array([
                [det.bbox.x1, det.bbox.y1],
                [det.bbox.x2, det.bbox.y1],
                [det.bbox.x2, det.bbox.y2],
                [det.bbox.x1, det.bbox.y2]
            ], dtype=np.float32)
            
            # Apply inverse transform
            ones = np.ones((4, 1))
            corners_homogeneous = np.hstack([corners, ones])
            transformed_corners = corners_homogeneous @ inv_transform.T
            
            # Get new bbox from transformed corners
            new_x1 = max(0, transformed_corners[:, 0].min())
            new_y1 = max(0, transformed_corners[:, 1].min())
            new_x2 = transformed_corners[:, 0].max()
            new_y2 = transformed_corners[:, 1].max()
            
            # Create compensated detection
            compensated_det = Detection(
                bbox=BoundingBox(
                    x1=float(new_x1),
                    y1=float(new_y1),
                    x2=float(new_x2),
                    y2=float(new_y2)
                ),
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                frame_id=det.frame_id
            )
            compensated.append(compensated_det)
        
        return compensated
    
    def reset(self):
        """Reset motion estimator state."""
        self.prev_gray = None
        self.prev_keypoints = None


class ForkliftTracker:
    """
    Multi-object tracker for forklifts using ByteTrack algorithm.
    
    Assigns persistent IDs to detected objects across frames and maintains
    track history for velocity estimation and state classification.
    
    Includes Camera Motion Compensation (CMC) to handle moving cameras.
    When the camera moves, CMC estimates the global motion and adjusts
    detection positions to maintain stable track IDs.
    
    Args:
        track_activation_threshold: Confidence threshold to create new tracks.
        lost_track_buffer: Frames to wait before deleting lost track.
        minimum_matching_threshold: IoU threshold for matching detections to tracks.
        frame_rate: Video frame rate for ByteTrack internals.
        enable_cmc: Enable Camera Motion Compensation for moving cameras.
        
    Example:
        >>> tracker = ForkliftTracker(enable_cmc=True)
        >>> for frame_id, (frame, detections) in enumerate(frames_and_detections):
        ...     tracked_objects = tracker.update(detections, frame_id, frame=frame)
        ...     for obj in tracked_objects:
        ...         print(f"Track {obj.track_id}: {obj.latest_bbox}")
    """
    
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        enable_cmc: bool = True  # NEW: Enable camera motion compensation
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
        self.enable_cmc = enable_cmc
        
        # Initialize ByteTrack
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate
        )
        
        # Camera Motion Compensation
        self._cmc = CameraMotionCompensator() if enable_cmc else None
        
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
            f"matching={minimum_matching_threshold}, "
            f"cmc={'enabled' if enable_cmc else 'disabled'}"
        )
    
    def update(
        self,
        detections: list[Detection],
        frame_id: int,
        frame: Optional[np.ndarray] = None  # NEW: Optional frame for CMC
    ) -> list[TrackedObject]:
        """
        Update tracker with new detections and return tracked objects.
        
        Args:
            detections: List of Detection objects from current frame.
            frame_id: Current frame index.
            frame: Optional frame image for camera motion compensation.
                   Required if enable_cmc=True for best tracking performance.
            frame_id: Current frame index.
            frame: Optional frame image for camera motion compensation.
                   Required if enable_cmc=True for best tracking performance.
            
        Returns:
            List of TrackedObject with updated track IDs.
        """
        self._current_frame_id = frame_id
        
        if not detections:
            # Still update CMC with frame even without detections
            if self.enable_cmc and self._cmc is not None and frame is not None:
                self._cmc.estimate_motion(frame)
            self._active_track_ids = set()
            return []
        
        # Camera Motion Compensation
        compensated_detections = detections
        if self.enable_cmc and self._cmc is not None and frame is not None:
            transform = self._cmc.estimate_motion(frame)
            if transform is not None:
                # Compensate detections to account for camera motion
                compensated_detections = self._cmc.compensate_detections(detections, transform)
                logger.debug(f"Frame {frame_id}: CMC applied camera motion compensation")
        
        # Convert detections to supervision format (using compensated coords for tracking)
        sv_detections = self._detections_to_supervision(compensated_detections)
        
        # Run ByteTrack
        tracked_detections = self._tracker.update_with_detections(sv_detections)
        
        # Convert results back to TrackedObject (but use ORIGINAL detection coords for storage)
        tracked_objects = self._process_tracked_detections(
            tracked_detections, detections, compensated_detections, frame_id
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
        compensated_detections: list[Detection],
        frame_id: int
    ) -> list[TrackedObject]:
        """Process ByteTrack results and update TrackedObject storage.
        
        Uses compensated coordinates for track association but stores
        original detection coordinates in the track history.
        """
        tracked_objects = []
        new_active_ids = set()
        
        if tracked_detections.tracker_id is None:
            return tracked_objects
        
        for i, track_id in enumerate(tracked_detections.tracker_id):
            track_id = int(track_id)
            new_active_ids.add(track_id)
            
            # Get tracked bbox (from compensated coordinates)
            xyxy = tracked_detections.xyxy[i]
            confidence = tracked_detections.confidence[i] if tracked_detections.confidence is not None else 0.5
            class_id = tracked_detections.class_id[i] if tracked_detections.class_id is not None else 0
            
            # Find matching original detection (use compensated for matching, original for storage)
            original_det = self._find_matching_detection(xyxy, original_detections, compensated_detections)
            
            if original_det is not None:
                # Use original detection coordinates (not compensated)
                detection = Detection(
                    bbox=original_det.bbox,
                    class_id=original_det.class_id,
                    class_name=original_det.class_name,
                    confidence=original_det.confidence,
                    frame_id=frame_id
                )
                class_name = original_det.class_name
            else:
                # Fallback: use tracked coordinates
                class_name = self._find_class_name(xyxy, original_detections)
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
    
    def _find_matching_detection(
        self,
        xyxy: np.ndarray,
        original_detections: list[Detection],
        compensated_detections: list[Detection]
    ) -> Optional[Detection]:
        """Find original detection that matches the tracked compensated bbox."""
        if len(original_detections) != len(compensated_detections):
            return None
        
        # Find closest compensated detection
        min_dist = float('inf')
        best_idx = -1
        
        for idx, comp_det in enumerate(compensated_detections):
            dist = (
                (comp_det.bbox.x1 - xyxy[0]) ** 2 +
                (comp_det.bbox.y1 - xyxy[1]) ** 2 +
                (comp_det.bbox.x2 - xyxy[2]) ** 2 +
                (comp_det.bbox.y2 - xyxy[3]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        
        if best_idx >= 0 and best_idx < len(original_detections):
            return original_detections[best_idx]
        return None
    
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
        
        # Reset CMC
        if self._cmc is not None:
            self._cmc.reset()
            
        logger.info("Tracker reset")
    
    @property
    def track_count(self) -> int:
        """Total number of tracks created."""
        return len(self._tracks)
    
    @property
    def active_track_count(self) -> int:
        """Number of currently active tracks."""
        return len(self._active_track_ids)
