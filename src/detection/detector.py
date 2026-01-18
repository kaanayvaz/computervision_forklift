"""
YOLOv8 detection wrapper for forklift, pallet, and person detection.

Provides ForkliftDetector class that wraps Ultralytics YOLOv8 with:
- Configuration-based initialization
- Detection result conversion to Detection dataclass
- Batch processing support
"""

import numpy as np
from pathlib import Path
from typing import Optional

from core.entities import Detection, BoundingBox
from core.utils import get_logger, load_config, Timer

logger = get_logger(__name__)

# Lazy import for ultralytics to allow testing without full install
_YOLO = None


def _get_yolo():
    """Lazy load YOLO to avoid import errors during testing."""
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO
        _YOLO = YOLO
    return _YOLO


class ForkliftDetector:
    """
    YOLO-based object detector for forklifts, pallets, and people.
    
    Args:
        config_path: Path to inference.yaml configuration file.
        
    Example:
        >>> detector = ForkliftDetector("config/inference.yaml")
        >>> detections = detector.detect_frame(frame, frame_id=0)
        >>> for det in detections:
        ...     print(f"{det.class_name}: {det.confidence:.2f}")
    """
    
    # Class ID to name mapping (update based on your model)
    DEFAULT_CLASS_NAMES = {
        0: "forklift",
        1: "pallet", 
        2: "person",
    }
    
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        
        # Extract config values
        model_config = self.config.get("model", {})
        detection_config = self.config.get("detection", {})
        
        self.weights_path = model_config.get("weights_path", "yolov8s.pt")
        self.device = model_config.get("device", "cuda")
        self.half_precision = model_config.get("half_precision", True)
        
        self.confidence_threshold = detection_config.get("confidence_threshold", 0.5)
        self.iou_threshold = detection_config.get("iou_threshold", 0.45)
        
        # Build class name mapping from config
        self.class_names = {}
        classes_config = detection_config.get("classes", {})
        for class_name, class_id in classes_config.items():
            self.class_names[class_id] = class_name
        
        # Fallback to defaults if not configured
        if not self.class_names:
            self.class_names = self.DEFAULT_CLASS_NAMES.copy()
        
        # Initialize model
        self._model = None
        self._timer = Timer()
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLO model from weights."""
        logger.info(f"Loading YOLO model: {self.weights_path}")
        
        try:
            YOLO = _get_yolo()
            self._model = YOLO(self.weights_path)
            
            # Set device
            if self.device == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning("CUDA not available, falling back to CPU")
                        self.device = "cpu"
                except ImportError:
                    logger.warning("PyTorch not found, using CPU")
                    self.device = "cpu"
            
            logger.info(
                f"Model loaded: device={self.device}, "
                f"conf={self.confidence_threshold}, iou={self.iou_threshold}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def detect_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0
    ) -> list[Detection]:
        """
        Run detection on a single frame.
        
        Args:
            frame: BGR image as numpy array.
            frame_id: Frame index for tracking.
            
        Returns:
            List of Detection objects.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        
        self._timer.start()
        
        # Run inference
        results = self._model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            half=self.half_precision and self.device != "cpu",
            verbose=False
        )
        
        elapsed_ms = self._timer.stop() * 1000
        
        # Convert results to Detection objects
        detections = self._parse_results(results, frame_id)
        
        logger.debug(
            f"Frame {frame_id}: {len(detections)} detections in {elapsed_ms:.1f}ms"
        )
        
        return detections
    
    def detect_batch(
        self,
        frames: list[np.ndarray],
        start_frame_id: int = 0
    ) -> list[list[Detection]]:
        """
        Run detection on a batch of frames.
        
        Args:
            frames: List of BGR images.
            start_frame_id: Starting frame index.
            
        Returns:
            List of detection lists, one per frame.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")
        
        if not frames:
            return []
        
        self._timer.start()
        
        # Run batch inference
        results = self._model(
            frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            half=self.half_precision and self.device != "cpu",
            verbose=False
        )
        
        elapsed_ms = self._timer.stop() * 1000
        
        # Convert results for each frame
        all_detections = []
        for i, result in enumerate(results):
            frame_id = start_frame_id + i
            detections = self._parse_single_result(result, frame_id)
            all_detections.append(detections)
        
        total_dets = sum(len(d) for d in all_detections)
        logger.debug(
            f"Batch: {len(frames)} frames, {total_dets} detections in {elapsed_ms:.1f}ms"
        )
        
        return all_detections
    
    def _parse_results(
        self,
        results: list,
        frame_id: int
    ) -> list[Detection]:
        """Parse YOLO results list to Detection objects."""
        if not results:
            return []
        
        return self._parse_single_result(results[0], frame_id)
    
    def _parse_single_result(
        self,
        result,
        frame_id: int
    ) -> list[Detection]:
        """Parse single YOLO result to Detection objects."""
        detections = []
        
        if result.boxes is None:
            return detections
        
        boxes = result.boxes
        
        for i in range(len(boxes)):
            # Get bounding box (xyxy format)
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            
            # Get confidence and class
            conf = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            
            # Get class name
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            # Create detection
            detection = Detection(
                bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                class_id=class_id,
                class_name=class_name,
                confidence=conf,
                frame_id=frame_id
            )
            
            detections.append(detection)
        
        return detections
    
    def filter_by_class(
        self,
        detections: list[Detection],
        class_names: list[str]
    ) -> list[Detection]:
        """
        Filter detections to only include specified classes.
        
        Args:
            detections: List of Detection objects.
            class_names: List of class names to keep.
            
        Returns:
            Filtered list of detections.
        """
        return [d for d in detections if d.class_name in class_names]
    
    def get_forklifts(self, detections: list[Detection]) -> list[Detection]:
        """Get only forklift detections."""
        return self.filter_by_class(detections, ["forklift"])
    
    def get_pallets(self, detections: list[Detection]) -> list[Detection]:
        """Get only pallet detections."""
        return self.filter_by_class(detections, ["pallet"])
    
    def get_people(self, detections: list[Detection]) -> list[Detection]:
        """Get only person detections."""
        return self.filter_by_class(detections, ["person"])
    
    @property
    def avg_inference_time_ms(self) -> float:
        """Average inference time in milliseconds."""
        return self._timer.elapsed * 1000
