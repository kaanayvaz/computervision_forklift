"""
Visualization module for frame annotation.

Draws bounding boxes, track IDs, states, and status indicators on video frames.
"""

import cv2
import numpy as np
from typing import Optional

from core.entities import Detection, TrackedObject, ForkliftState, BoundingBox
from core.utils import get_logger

logger = get_logger(__name__)


# Color definitions (BGR format)
COLORS = {
    "forklift": (0, 255, 0),      # Green
    "pallet": (255, 165, 0),       # Orange
    "person": (0, 255, 255),       # Yellow
    "unknown": (128, 128, 128),    # Gray
    
    # State colors
    ForkliftState.IDLE: (0, 0, 255),           # Red
    ForkliftState.MOVING_EMPTY: (255, 255, 0), # Cyan
    ForkliftState.MOVING_LOADED: (0, 255, 0),  # Green
    ForkliftState.LOADING: (255, 0, 255),      # Magenta
    ForkliftState.UNLOADING: (255, 0, 255),    # Magenta
    ForkliftState.UNKNOWN: (128, 128, 128),    # Gray
}


class Visualizer:
    """
    Annotate video frames with detection and tracking information.
    
    Draws:
    - Bounding boxes with class-specific colors
    - Track IDs and confidence scores
    - State indicators (idle/active/carrying)
    - Velocity indicators
    
    Args:
        font_scale: Text size multiplier.
        line_thickness: Border line thickness.
        show_confidence: Whether to display confidence scores.
    """
    
    def __init__(
        self,
        font_scale: float = 0.6,
        line_thickness: int = 2,
        show_confidence: bool = True
    ):
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.show_confidence = show_confidence
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        tracks: list[TrackedObject],
        detections: Optional[list[Detection]] = None,
        frame_id: int = 0,
        show_pallets: bool = True
    ) -> np.ndarray:
        """
        Annotate frame with tracking visualization.
        
        Args:
            frame: Input frame (BGR).
            tracks: List of tracked objects.
            detections: Optional list of detections (for untracked objects).
            frame_id: Current frame number.
            show_pallets: Whether to show pallet detections.
            
        Returns:
            Annotated frame.
        """
        annotated = frame.copy()
        
        # Draw tracked forklifts
        for track in tracks:
            if track.class_name == "forklift":
                self._draw_forklift(annotated, track)
            elif track.class_name == "pallet" and show_pallets:
                self._draw_pallet(annotated, track)
        
        # Draw untracked detections
        if detections:
            for det in detections:
                if det.class_name == "pallet" and show_pallets:
                    self._draw_detection(annotated, det)
        
        # Draw frame info
        self._draw_frame_info(annotated, frame_id, len(tracks))
        
        return annotated
    
    def _draw_forklift(
        self,
        frame: np.ndarray,
        track: TrackedObject
    ) -> None:
        """Draw forklift with state visualization."""
        bbox = track.latest_bbox
        if bbox is None:
            return
        
        # Get color based on state
        state_color = COLORS.get(track.current_state, COLORS["unknown"])
        
        # Draw bounding box
        pt1 = (int(bbox.x1), int(bbox.y1))
        pt2 = (int(bbox.x2), int(bbox.y2))
        cv2.rectangle(frame, pt1, pt2, state_color, self.line_thickness)
        
        # Draw label background
        label = f"#{track.track_id} {track.current_state.value}"
        if track.is_carrying_pallet:
            label += " [LOADED]"
        
        label_size, _ = cv2.getTextSize(label, self.font, self.font_scale, 1)
        label_bg_pt1 = (int(bbox.x1), int(bbox.y1) - label_size[1] - 10)
        label_bg_pt2 = (int(bbox.x1) + label_size[0] + 5, int(bbox.y1))
        
        cv2.rectangle(frame, label_bg_pt1, label_bg_pt2, state_color, -1)
        
        # Draw label text
        cv2.putText(
            frame, label,
            (int(bbox.x1) + 2, int(bbox.y1) - 5),
            self.font, self.font_scale, (0, 0, 0), 1
        )
        
        # Draw velocity indicator
        if track.velocity > 2.0:
            self._draw_velocity_arrow(frame, bbox, track.velocity)
    
    def _draw_velocity_arrow(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        velocity: float
    ) -> None:
        """Draw velocity arrow indicator."""
        center = (int(bbox.centroid_x), int(bbox.centroid_y))
        
        # Simple horizontal arrow for now (direction could be added)
        arrow_length = min(int(velocity * 3), 50)
        end_point = (center[0] + arrow_length, center[1])
        
        cv2.arrowedLine(frame, center, end_point, (255, 255, 255), 2, tipLength=0.3)
    
    def _draw_pallet(
        self,
        frame: np.ndarray,
        track: TrackedObject
    ) -> None:
        """Draw pallet detection."""
        bbox = track.latest_bbox
        if bbox is None:
            return
        
        color = COLORS["pallet"]
        pt1 = (int(bbox.x1), int(bbox.y1))
        pt2 = (int(bbox.x2), int(bbox.y2))
        
        cv2.rectangle(frame, pt1, pt2, color, self.line_thickness)
        
        label = f"Pallet #{track.track_id}"
        cv2.putText(
            frame, label,
            (int(bbox.x1), int(bbox.y1) - 5),
            self.font, self.font_scale * 0.8, color, 1
        )
    
    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: Detection
    ) -> None:
        """Draw raw detection (untracked)."""
        bbox = detection.bbox
        color = COLORS.get(detection.class_name, COLORS["unknown"])
        
        pt1 = (int(bbox.x1), int(bbox.y1))
        pt2 = (int(bbox.x2), int(bbox.y2))
        
        cv2.rectangle(frame, pt1, pt2, color, 1)  # Thinner for untracked
        
        if self.show_confidence:
            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.putText(
                frame, label,
                (int(bbox.x1), int(bbox.y1) - 5),
                self.font, self.font_scale * 0.7, color, 1
            )
    
    def _draw_frame_info(
        self,
        frame: np.ndarray,
        frame_id: int,
        track_count: int
    ) -> None:
        """Draw frame information overlay."""
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Frame info text
        cv2.putText(
            frame, f"Frame: {frame_id}",
            (20, 35), self.font, 0.6, (255, 255, 255), 1
        )
        cv2.putText(
            frame, f"Tracks: {track_count}",
            (20, 55), self.font, 0.6, (255, 255, 255), 1
        )
    
    def draw_state_indicator(
        self,
        frame: np.ndarray,
        track: TrackedObject,
        position: str = "top"
    ) -> np.ndarray:
        """
        Draw prominent state indicator.
        
        Args:
            frame: Input frame.
            track: TrackedObject to display.
            position: "top" or "bottom" for indicator placement.
            
        Returns:
            Frame with indicator.
        """
        bbox = track.latest_bbox
        if bbox is None:
            return frame
        
        state = track.current_state
        color = COLORS.get(state, COLORS["unknown"])
        
        # Draw state indicator circle
        if position == "top":
            center = (int(bbox.centroid_x), int(bbox.y1) - 15)
        else:
            center = (int(bbox.centroid_x), int(bbox.y2) + 15)
        
        cv2.circle(frame, center, 10, color, -1)
        cv2.circle(frame, center, 10, (255, 255, 255), 2)
        
        return frame
    
    def create_legend(
        self,
        width: int = 200,
        height: int = 150
    ) -> np.ndarray:
        """Create legend image for state colors."""
        legend = np.zeros((height, width, 3), dtype=np.uint8)
        legend.fill(30)  # Dark gray background
        
        y_offset = 20
        states = [
            (ForkliftState.IDLE, "Idle"),
            (ForkliftState.MOVING_EMPTY, "Moving (Empty)"),
            (ForkliftState.MOVING_LOADED, "Moving (Loaded)"),
            (ForkliftState.LOADING, "Loading"),
        ]
        
        for state, label in states:
            color = COLORS.get(state, COLORS["unknown"])
            cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 5), color, -1)
            cv2.putText(legend, label, (40, y_offset), self.font, 0.5, (255, 255, 255), 1)
            y_offset += 30
        
        return legend
