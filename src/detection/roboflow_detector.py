"""
Roboflow cloud-based detector for forklift and pallet detection.

Uses Roboflow's hosted inference API for video processing.
This eliminates the need for local GPU training.
"""

import os
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from core.entities import Detection, BoundingBox
from core.utils import get_logger
from core.env_config import get_roboflow_config

logger = get_logger(__name__)


@dataclass
class RoboflowConfig:
    """Configuration for Roboflow API."""
    api_key: str
    forklift_project: str = "forklift-0jmzj-uvcoy"
    forklift_version: int = 1
    pallet_project: str = "pallet-6awi8-zcqu2"
    pallet_version: int = 1
    workspace: str = "ss-rz6v8"


class RoboflowDetector:
    """
    Roboflow cloud-based detector for video inference.
    
    Uses Roboflow's hosted models for forklift and pallet detection.
    Supports both single-frame and batch video inference.
    
    Example:
        >>> # Load from environment
        >>> detector = RoboflowDetector.from_env()
        >>> 
        >>> # Or pass API key directly
        >>> detector = RoboflowDetector(api_key="your_key")
        >>> results = detector.process_video("video.mp4", fps=5)
    """
    
    # Class name mappings
    CLASS_NAMES = {
        "forklift": 0,
        "Forklift - v1 2023-05-24 1-43pm": 0,  # Roboflow class name
        "pallet": 1,
        "wood pallet": 1,  # Roboflow class name
        "person": 2,
    }
    
    def __init__(
        self,
        api_key: str,
        config: Optional[RoboflowConfig] = None
    ):
        """
        Initialize Roboflow detector.
        
        Args:
            api_key: Roboflow API key
            config: Optional configuration object
        """
        self.api_key = api_key
        self.config = config or RoboflowConfig(api_key=api_key)
        
        self._rf = None
        self._forklift_model = None
        self._pallet_model = None
        
        self._init_models()
    
    @classmethod
    def from_env(cls) -> "RoboflowDetector":
        """
        Create RoboflowDetector from environment variables.
        
        Loads configuration from .env file using ROBOFLOW_* variables.
        
        Returns:
            Configured RoboflowDetector instance.
            
        Raises:
            ValueError: If required environment variables are missing.
        """
        env_config = get_roboflow_config()
        
        config = RoboflowConfig(
            api_key=env_config["api_key"],
            workspace=env_config["workspace"],
            forklift_project=env_config["forklift_project"],
            forklift_version=env_config["forklift_version"],
            pallet_project=env_config["pallet_project"],
            pallet_version=env_config["pallet_version"],
        )
        
        return cls(api_key=env_config["api_key"], config=config)
    
    def _init_models(self):
        """Initialize Roboflow models."""
        try:
            from roboflow import Roboflow
            
            if not self.api_key:
                raise ValueError(
                    "Roboflow API key is required. "
                    "Set ROBOFLOW_API_KEY in .env file or pass api_key parameter."
                )
            
            logger.info("Initializing Roboflow API...")
            self._rf = Roboflow(api_key=self.api_key)
            
            # Load forklift model
            logger.info(f"Loading forklift model: {self.config.forklift_project}")
            forklift_project = self._rf.workspace(self.config.workspace).project(
                self.config.forklift_project
            )
            self._forklift_model = forklift_project.version(
                self.config.forklift_version
            ).model
            
            # Load pallet model
            logger.info(f"Loading pallet model: {self.config.pallet_project}")
            pallet_project = self._rf.workspace(self.config.workspace).project(
                self.config.pallet_project
            )
            self._pallet_model = pallet_project.version(
                self.config.pallet_version
            ).model
            
            logger.info("Roboflow models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Roboflow: {e}")
            raise
    
    def detect_frame(
        self,
        image_path: str,
        confidence: float = 0.25,
        frame_id: int = 0
    ) -> list[Detection]:
        """
        Run detection on a single image.
        
        Args:
            image_path: Path to image file
            confidence: Minimum confidence threshold
            frame_id: Frame ID for tracking
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        # Forklift detection
        try:
            forklift_result = self._forklift_model.predict(
                image_path,
                confidence=int(confidence * 100)
            ).json()
            
            for pred in forklift_result.get("predictions", []):
                det = self._parse_prediction(pred, "forklift", frame_id)
                if det:
                    detections.append(det)
                    
        except Exception as e:
            logger.warning(f"Forklift detection failed: {e}")
        
        # Pallet detection
        try:
            pallet_result = self._pallet_model.predict(
                image_path,
                confidence=int(confidence * 100)
            ).json()
            
            for pred in pallet_result.get("predictions", []):
                det = self._parse_prediction(pred, "pallet", frame_id)
                if det:
                    detections.append(det)
                    
        except Exception as e:
            logger.warning(f"Pallet detection failed: {e}")
        
        return detections
    
    def process_video(
        self,
        video_path: str,
        fps: int = 5,
        confidence: float = 0.25,
        output_dir: Optional[str] = None
    ) -> dict:
        """
        Process video using Roboflow batch inference.
        
        Args:
            video_path: Path to input video
            fps: Frames per second to process
            confidence: Minimum confidence threshold
            output_dir: Optional output directory for results
            
        Returns:
            Dictionary with detection results and metadata
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"FPS: {fps}, Confidence: {confidence}")
        
        results = {
            "video_path": str(video_path),
            "fps": fps,
            "forklift_detections": [],
            "pallet_detections": [],
        }
        
        # Process with forklift model
        logger.info("Running forklift detection...")
        try:
            job_id, signed_url, expire_time = self._forklift_model.predict_video(
                str(video_path),
                fps=fps,
                prediction_type="batch-video",
            )
            
            logger.info(f"Forklift job started: {job_id}")
            forklift_results = self._forklift_model.poll_until_video_results(job_id)
            results["forklift_detections"] = forklift_results
            logger.info(f"Forklift detection complete")
            
        except Exception as e:
            logger.error(f"Forklift video processing failed: {e}")
            results["forklift_error"] = str(e)
        
        # Process with pallet model
        logger.info("Running pallet detection...")
        try:
            job_id, signed_url, expire_time = self._pallet_model.predict_video(
                str(video_path),
                fps=fps,
                prediction_type="batch-video",
            )
            
            logger.info(f"Pallet job started: {job_id}")
            pallet_results = self._pallet_model.poll_until_video_results(job_id)
            results["pallet_detections"] = pallet_results
            logger.info(f"Pallet detection complete")
            
        except Exception as e:
            logger.error(f"Pallet video processing failed: {e}")
            results["pallet_error"] = str(e)
        
        # Save results
        if output_dir:
            output_path = Path(output_dir) / f"{video_path.stem}_roboflow_results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to: {output_path}")
            results["output_path"] = str(output_path)
        
        return results
    
    def _parse_prediction(
        self,
        pred: dict,
        class_name: str,
        frame_id: int
    ) -> Optional[Detection]:
        """Parse Roboflow prediction to Detection object."""
        try:
            # Roboflow uses center coordinates
            x = pred.get("x", 0)
            y = pred.get("y", 0)
            width = pred.get("width", 0)
            height = pred.get("height", 0)
            
            # Convert to corner coordinates
            x1 = x - width / 2
            y1 = y - height / 2
            x2 = x + width / 2
            y2 = y + height / 2
            
            confidence = pred.get("confidence", 0)
            
            class_id = self.CLASS_NAMES.get(class_name, 0)
            
            return Detection(
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                frame_id=frame_id
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse prediction: {e}")
            return None
    
    def convert_results_to_detections(
        self,
        results: dict,
        fps: int = 5
    ) -> dict[int, list[Detection]]:
        """
        Convert Roboflow video results to frame-indexed detections.
        
        Args:
            results: Results from process_video()
            fps: FPS used during processing
            
        Returns:
            Dictionary mapping frame_id to list of Detection objects
        """
        frame_detections = {}
        
        # Process forklift detections
        forklift_data = results.get("forklift_detections", {})
        if isinstance(forklift_data, dict):
            for frame_key, frame_data in forklift_data.items():
                try:
                    frame_id = int(frame_key)
                except ValueError:
                    continue
                    
                if frame_id not in frame_detections:
                    frame_detections[frame_id] = []
                
                predictions = frame_data if isinstance(frame_data, list) else frame_data.get("predictions", [])
                for pred in predictions:
                    det = self._parse_prediction(pred, "forklift", frame_id)
                    if det:
                        frame_detections[frame_id].append(det)
        
        # Process pallet detections
        pallet_data = results.get("pallet_detections", {})
        if isinstance(pallet_data, dict):
            for frame_key, frame_data in pallet_data.items():
                try:
                    frame_id = int(frame_key)
                except ValueError:
                    continue
                    
                if frame_id not in frame_detections:
                    frame_detections[frame_id] = []
                
                predictions = frame_data if isinstance(frame_data, list) else frame_data.get("predictions", [])
                for pred in predictions:
                    det = self._parse_prediction(pred, "pallet", frame_id)
                    if det:
                        frame_detections[frame_id].append(det)
        
        return frame_detections
