"""
Video reader module for frame extraction.

Provides VideoReader class for iterating over video frames with:
- Frame skipping for performance
- Resolution limiting
- Metadata extraction
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Optional, Tuple

from core.utils import get_logger

logger = get_logger(__name__)


class VideoReader:
    """
    Video reader with frame extraction and preprocessing.
    
    Supports frame skipping and resolution limiting for performance optimization.
    
    Args:
        video_path: Path to video file.
        frame_skip: Process every Nth frame (1 = all frames).
        max_resolution: Maximum (width, height) for frames. None = no limit.
        
    Example:
        >>> reader = VideoReader("video.mp4", frame_skip=2)
        >>> for frame_id, frame in reader:
        ...     process(frame)
    """
    
    def __init__(
        self,
        video_path: str | Path,
        frame_skip: int = 1,
        max_resolution: Optional[Tuple[int, int]] = None
    ):
        self.video_path = Path(video_path)
        self.frame_skip = max(1, frame_skip)
        self.max_resolution = max_resolution
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[dict] = None
        
        # Validate file exists
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        # Initialize capture and get metadata
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize video capture and extract metadata."""
        self._cap = cv2.VideoCapture(str(self.video_path))
        
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        # Extract metadata
        self._metadata = {
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "codec": int(self._cap.get(cv2.CAP_PROP_FOURCC)),
        }
        
        # Calculate duration
        if self._metadata["fps"] > 0:
            self._metadata["duration_seconds"] = (
                self._metadata["total_frames"] / self._metadata["fps"]
            )
        else:
            self._metadata["duration_seconds"] = 0.0
        
        # Calculate effective FPS after skip
        self._metadata["effective_fps"] = self._metadata["fps"] / self.frame_skip
        
        logger.info(
            f"Opened video: {self.video_path.name} "
            f"({self._metadata['width']}x{self._metadata['height']}, "
            f"{self._metadata['fps']:.1f} fps, "
            f"{self._metadata['duration_seconds']:.1f}s)"
        )
    
    @property
    def fps(self) -> float:
        """Video frames per second."""
        return self._metadata["fps"] if self._metadata else 0.0
    
    @property
    def effective_fps(self) -> float:
        """Effective FPS after frame skipping."""
        return self._metadata["effective_fps"] if self._metadata else 0.0
    
    @property
    def total_frames(self) -> int:
        """Total number of frames in video."""
        return self._metadata["total_frames"] if self._metadata else 0
    
    @property
    def duration_seconds(self) -> float:
        """Video duration in seconds."""
        return self._metadata["duration_seconds"] if self._metadata else 0.0
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Video resolution as (width, height)."""
        if self._metadata:
            return (self._metadata["width"], self._metadata["height"])
        return (0, 0)
    
    def get_metadata(self) -> dict:
        """
        Get video metadata.
        
        Returns:
            Dictionary with: fps, width, height, total_frames, 
            duration_seconds, effective_fps
        """
        return self._metadata.copy() if self._metadata else {}
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if it exceeds max_resolution."""
        if self.max_resolution is None:
            return frame
        
        h, w = frame.shape[:2]
        max_w, max_h = self.max_resolution
        
        if w <= max_w and h <= max_h:
            return frame
        
        # Calculate scale factor to fit within max_resolution
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Iterate over video frames.
        
        Yields:
            Tuple of (frame_id, frame) where frame is BGR numpy array.
        """
        if self._cap is None:
            raise RuntimeError("Video capture not initialized")
        
        # Reset to beginning
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_id = 0
        
        while True:
            ret, frame = self._cap.read()
            
            if not ret:
                break
            
            # Apply frame skipping
            if frame_id % self.frame_skip == 0:
                # Resize if needed
                frame = self._resize_frame(frame)
                yield frame_id, frame
            
            frame_id += 1
    
    def read_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Read a specific frame by ID.
        
        Args:
            frame_id: Frame index to read.
            
        Returns:
            Frame as BGR numpy array, or None if unavailable.
        """
        if self._cap is None:
            return None
        
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self._cap.read()
        
        if ret:
            return self._resize_frame(frame)
        return None
    
    def release(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug(f"Released video: {self.video_path.name}")
    
    def __enter__(self) -> "VideoReader":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - release resources."""
        self.release()
    
    def __del__(self) -> None:
        """Destructor - ensure resources are released."""
        self.release()


class VideoWriter:
    """
    Video writer for saving annotated frames.
    
    Args:
        output_path: Path for output video file.
        fps: Frames per second.
        resolution: Video resolution as (width, height).
        codec: FourCC codec string (default: "mp4v").
    """
    
    def __init__(
        self,
        output_path: str | Path,
        fps: float,
        resolution: Tuple[int, int],
        codec: str = "mp4v"
    ):
        self.output_path = Path(output_path)
        self.fps = fps
        self.resolution = resolution
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            resolution
        )
        
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {self.output_path}")
        
        self._frame_count = 0
        logger.info(f"Created video writer: {self.output_path.name}")
    
    def write(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video.
        
        Args:
            frame: BGR frame to write.
        """
        # Resize if needed
        h, w = frame.shape[:2]
        if (w, h) != self.resolution:
            frame = cv2.resize(frame, self.resolution)
        
        self._writer.write(frame)
        self._frame_count += 1
    
    @property
    def frame_count(self) -> int:
        """Number of frames written."""
        return self._frame_count
    
    def release(self) -> None:
        """Release video writer resources."""
        if self._writer is not None:
            self._writer.release()
            logger.info(f"Saved video: {self.output_path.name} ({self._frame_count} frames)")
    
    def __enter__(self) -> "VideoWriter":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - release resources."""
        self.release()
