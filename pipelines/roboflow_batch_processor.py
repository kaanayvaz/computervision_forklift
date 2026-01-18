"""
Roboflow-integrated batch processing pipeline for video analysis.

This pipeline uses Roboflow cloud inference for forklift and pallet detection,
then processes the results through the full analytics pipeline:
1. Roboflow cloud detection (forklift + pallet)
2. Object tracking (ByteTrack)
3. Spatial analysis (pallet carrying detection)
4. Motion estimation
5. State classification
6. Activity segmentation
7. Analytics generation
8. Report output
"""

import sys
from pathlib import Path

# Add src to path if needed
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import json
import cv2
import numpy as np
from typing import Optional
from datetime import datetime
from tqdm import tqdm

from core.entities import TrackedObject, ForkliftState, Detection, BoundingBox
from core.utils import get_logger, load_config, FPSTracker
from core.env_config import get_roboflow_config
from video.reader import VideoReader, VideoWriter
from tracking.tracker import ForkliftTracker
from spatial.pallet_detector import SpatialAnalyzer
from motion.motion_estimator import MotionEstimator
from state.classifier import StateClassifier
from analytics.activity_segmenter import ActivitySegmenter
from analytics.metrics import generate_analytics, generate_summary_report
from analytics.reporter import Reporter
from visualization.visualizer import Visualizer

logger = get_logger(__name__)


class RoboflowBatchProcessor:
    """
    End-to-end batch processing pipeline using Roboflow cloud detection.
    
    This processor:
    1. Submits video to Roboflow for forklift and pallet detection
    2. Processes detection results through tracking pipeline
    3. Performs spatial analysis to detect pallet carrying
    4. Classifies forklift states (IDLE, MOVING_EMPTY, MOVING_LOADED, etc.)
    5. Generates comprehensive analytics and reports
    
    Args:
        config_dir: Path to configuration directory.
        output_dir: Path for output files.
        visualize: Whether to generate annotated output video.
        fps: Frames per second for Roboflow processing.
        confidence: Detection confidence threshold.
    """
    
    def __init__(
        self,
        config_dir: str | Path = "config",
        output_dir: str | Path = "data/outputs",
        visualize: bool = True,
        fps: int = 5,
        confidence: float = 0.25
    ):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.visualize = visualize
        self.fps = fps
        self.confidence = confidence
        
        # Load configurations
        self.rules_config = load_config(self.config_dir / "rules.yaml")
        
        # Roboflow configuration
        self.roboflow_config = get_roboflow_config()
        
        # Initialize Roboflow
        self._rf = None
        self._forklift_model = None
        self._pallet_model = None
        
        # Initialize pipeline components
        self._init_components()
        
        logger.info(f"RoboflowBatchProcessor initialized: config={config_dir}, output={output_dir}")
    
    def _init_roboflow(self):
        """Initialize Roboflow API and models."""
        try:
            from roboflow import Roboflow
            
            logger.info("Initializing Roboflow API...")
            self._rf = Roboflow(api_key=self.roboflow_config["api_key"])
            
            # Load forklift model
            workspace = self.roboflow_config.get("workspace")
            
            logger.info(f"Loading forklift model: {self.roboflow_config['forklift_project']}")
            if workspace:
                forklift_project = self._rf.workspace(workspace).project(
                    self.roboflow_config["forklift_project"]
                )
            else:
                forklift_project = self._rf.workspace().project(
                    self.roboflow_config["forklift_project"]
                )
            self._forklift_model = forklift_project.version(
                self.roboflow_config["forklift_version"]
            ).model
            
            # Load pallet model
            logger.info(f"Loading pallet model: {self.roboflow_config['pallet_project']}")
            if workspace:
                pallet_project = self._rf.workspace(workspace).project(
                    self.roboflow_config["pallet_project"]
                )
            else:
                pallet_project = self._rf.workspace().project(
                    self.roboflow_config["pallet_project"]
                )
            self._pallet_model = pallet_project.version(
                self.roboflow_config["pallet_version"]
            ).model
            
            logger.info("Roboflow models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Roboflow: {e}")
            raise
    
    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        # Tracking
        self.tracker = ForkliftTracker()
        
        # Spatial analysis
        self.spatial_analyzer = SpatialAnalyzer(self.rules_config)
        
        # Motion estimation
        motion_config = self.rules_config.get("motion", {})
        self.motion_estimator = MotionEstimator(
            smoothing_window=motion_config.get("smoothing_window", 5)
        )
        
        # State classification
        state_config = self.rules_config.get("state", {})
        self.state_classifier = StateClassifier(
            idle_threshold=motion_config.get("velocity_idle_threshold", 2.0),
            confirmation_frames=state_config.get("state_confirmation_frames", 5)
        )
        
        # Activity segmentation
        activity_config = self.rules_config.get("activity", {})
        self.activity_segmenter = ActivitySegmenter(
            min_duration=activity_config.get("min_duration", 5.0)
        )
        
        # Visualization
        self.visualizer = Visualizer()
        
        # Reporting
        self.reporter = Reporter(self.output_dir / "reports")
        
        # FPS tracking
        self.fps_tracker = FPSTracker()
    
    def process_video(
        self,
        video_path: str | Path,
        output_name: Optional[str] = None
    ) -> dict:
        """
        Process a video file through the complete Roboflow + Analytics pipeline.
        
        Args:
            video_path: Path to input video.
            output_name: Optional name for output files.
            
        Returns:
            Dictionary with processing results and analytics.
        """
        video_path = Path(video_path)
        if output_name is None:
            output_name = video_path.stem
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print("\n" + "=" * 60)
        print("ROBOFLOW INTEGRATED PIPELINE")
        print("=" * 60)
        print(f"Video: {video_path}")
        print(f"FPS: {self.fps}")
        print(f"Confidence: {self.confidence}")
        print("=" * 60 + "\n")
        
        # Initialize Roboflow if not already done
        if self._rf is None:
            self._init_roboflow()
        
        # Reset pipeline components
        self.tracker.reset()
        self.motion_estimator.reset()
        self.state_classifier.reset()
        
        # Step 1: Run Roboflow cloud detection
        logger.info("Step 1: Running Roboflow cloud detection...")
        roboflow_results = self._run_roboflow_detection(str(video_path))
        
        # Step 2: Convert results to Detection objects
        logger.info("Step 2: Converting detection results...")
        frame_detections = self._convert_roboflow_results(roboflow_results)
        
        # Get video metadata
        reader = VideoReader(video_path)
        video_fps = reader.fps
        total_frames = reader.total_frames
        duration_seconds = reader.duration_seconds
        resolution = reader.resolution
        
        # Calculate frame mapping (Roboflow processes at self.fps, video is at video_fps)
        frame_interval = int(video_fps / self.fps) if self.fps < video_fps else 1
        
        # Step 3: Process detections through analytics pipeline
        logger.info("Step 3: Running tracking and analytics pipeline...")
        all_tracks, processed_frames = self._process_detections(
            frame_detections, 
            frame_interval,
            total_frames
        )
        
        # Step 4: Generate visualization if enabled
        if self.visualize:
            logger.info("Step 4: Generating annotated video...")
            self._generate_visualization(
                video_path, 
                all_tracks, 
                frame_detections,
                frame_interval,
                output_name
            )
        
        # Step 5: Generate analytics and reports
        logger.info("Step 5: Generating analytics and reports...")
        results = self._generate_results(
            all_tracks, 
            duration_seconds, 
            output_name,
            roboflow_results
        )
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Forklifts Tracked: {results['track_count']}")
        print(f"Frames Processed: {processed_frames}")
        print(f"Total Duration: {duration_seconds:.1f} seconds")
        print(f"Utilization: {results['analytics'].utilization_percentage:.1f}%")
        print(f"Idle Time: {results['analytics'].total_idle_time_seconds / 60:.1f} minutes")
        print(f"Cost of Waste: ${results['analytics'].cost_of_waste:.2f}")
        print("=" * 60)
        print(f"\nReports saved to: {self.output_dir}/reports/")
        if self.visualize:
            print(f"Annotated video saved to: {self.output_dir}/videos/")
        
        return results
    
    def _run_roboflow_detection(self, video_path: str) -> dict:
        """Run Roboflow batch video detection for forklifts and pallets."""
        results = {
            "forklift_detections": {},
            "pallet_detections": {},
            "fps": self.fps,
            "confidence": self.confidence
        }
        
        # Run forklift detection
        print("\n📦 Running forklift detection...")
        try:
            job_id, signed_url, expire_time = self._forklift_model.predict_video(
                video_path,
                fps=self.fps,
                prediction_type="batch-video",
            )
            logger.info(f"Forklift job submitted: {job_id}")
            print(f"   Job ID: {job_id}")
            print("   Waiting for results (this may take a few minutes)...")
            
            forklift_results = self._forklift_model.poll_until_video_results(job_id)
            results["forklift_detections"] = forklift_results
            
            # Count detections
            total = sum(len(f.get("predictions", [])) if isinstance(f, dict) else len(f) 
                       for f in forklift_results.values() if f)
            print(f"   ✅ Forklift detection complete: {total} detections")
            
        except Exception as e:
            logger.error(f"Forklift detection failed: {e}")
            results["forklift_error"] = str(e)
            print(f"   ❌ Forklift detection failed: {e}")
        
        # Run pallet detection
        print("\n📦 Running pallet detection...")
        try:
            job_id, signed_url, expire_time = self._pallet_model.predict_video(
                video_path,
                fps=self.fps,
                prediction_type="batch-video",
            )
            logger.info(f"Pallet job submitted: {job_id}")
            print(f"   Job ID: {job_id}")
            print("   Waiting for results (this may take a few minutes)...")
            
            pallet_results = self._pallet_model.poll_until_video_results(job_id)
            results["pallet_detections"] = pallet_results
            
            # Count detections
            total = sum(len(f.get("predictions", [])) if isinstance(f, dict) else len(f) 
                       for f in pallet_results.values() if f)
            print(f"   ✅ Pallet detection complete: {total} detections")
            
        except Exception as e:
            logger.error(f"Pallet detection failed: {e}")
            results["pallet_error"] = str(e)
            print(f"   ❌ Pallet detection failed: {e}")
        
        return results
    
    def _convert_roboflow_results(self, results: dict) -> dict[int, list[Detection]]:
        """Convert Roboflow results to frame-indexed Detection objects."""
        frame_detections = {}
        
        # Process forklift detections
        forklift_data = results.get("forklift_detections", {})
        self._process_roboflow_model_results(forklift_data, "forklift", frame_detections)
        
        # Process pallet detections
        pallet_data = results.get("pallet_detections", {})
        self._process_roboflow_model_results(pallet_data, "pallet", frame_detections)
        
        logger.info(f"Converted {len(frame_detections)} frames with detections")
        return frame_detections
    
    def _process_roboflow_model_results(
        self, 
        model_data: dict, 
        class_name: str, 
        frame_detections: dict
    ) -> None:
        """
        Process Roboflow batch video results for a single model.
        
        Roboflow batch video API returns data in this format:
        {
            "frame_offset": [0, 12, 25, ...],  # Frame numbers
            "time_offset": [0.0, 0.48, ...],   # Time in seconds
            "project-name": [                   # Array of frame results
                {
                    "predictions": [...],
                    "image": {...}
                },
                ...
            ]
        }
        """
        if not isinstance(model_data, dict):
            return
        
        # Get frame offsets
        frame_offsets = model_data.get("frame_offset", [])
        
        # Find the project key (contains the actual predictions)
        project_key = None
        for key in model_data.keys():
            if key not in ["frame_offset", "time_offset", "fps", "confidence"]:
                project_key = key
                break
        
        if project_key is None:
            # Try alternative format: direct frame indexing
            for frame_key, frame_data in model_data.items():
                try:
                    frame_id = int(frame_key)
                except (ValueError, TypeError):
                    continue
                
                if frame_id not in frame_detections:
                    frame_detections[frame_id] = []
                
                predictions = frame_data if isinstance(frame_data, list) else frame_data.get("predictions", [])
                for pred in predictions:
                    det = self._parse_prediction(pred, class_name, frame_id)
                    if det:
                        frame_detections[frame_id].append(det)
            return
        
        # Process the project results (array of frame data)
        frame_results = model_data.get(project_key, [])
        
        for i, frame_data in enumerate(frame_results):
            # Get frame ID from frame_offsets or use index
            if i < len(frame_offsets):
                frame_id = frame_offsets[i]
            else:
                frame_id = i
            
            if frame_id not in frame_detections:
                frame_detections[frame_id] = []
            
            # Get predictions from frame data
            predictions = []
            if isinstance(frame_data, dict):
                predictions = frame_data.get("predictions", [])
            elif isinstance(frame_data, list):
                predictions = frame_data
            
            for pred in predictions:
                det = self._parse_prediction(pred, class_name, frame_id)
                if det:
                    frame_detections[frame_id].append(det)
    
    def _parse_prediction(self, pred: dict, class_name: str, frame_id: int) -> Optional[Detection]:
        """Parse a Roboflow prediction into a Detection object."""
        try:
            # Roboflow uses center coordinates
            x = pred.get("x", 0)
            y = pred.get("y", 0)
            width = pred.get("width", 0)
            height = pred.get("height", 0)
            
            # Filter out invalid or unreasonably large detections
            # Typical forklift size: 100-400px width/height
            # Typical pallet size: 50-300px width/height
            max_size = 500 if class_name == "forklift" else 400
            min_size = 30 if class_name == "forklift" else 20
            
            if width > max_size or height > max_size:
                logger.debug(f"Skipping oversized detection: {class_name} {width}x{height}")
                return None
            
            if width < min_size or height < min_size:
                logger.debug(f"Skipping undersized detection: {class_name} {width}x{height}")
                return None
            
            # Filter low confidence
            confidence = pred.get("confidence", 0)
            min_confidence = 0.4 if class_name == "forklift" else 0.3
            if confidence < min_confidence:
                return None
            
            # Convert to corner coordinates
            x1 = x - width / 2
            y1 = y - height / 2
            x2 = x + width / 2
            y2 = y + height / 2
            
            class_id = 0 if class_name == "forklift" else 1
            
            return Detection(
                bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidence),
                frame_id=frame_id
            )
        except Exception as e:
            logger.warning(f"Failed to parse prediction: {e}")
            return None
    
    def _process_detections(
        self, 
        frame_detections: dict[int, list[Detection]],
        frame_interval: int,
        total_frames: int
    ) -> tuple[dict[int, TrackedObject], int]:
        """Process frame detections through tracking and state classification."""
        all_tracks: dict[int, TrackedObject] = {}
        processed_frames = 0
        
        # Sort frame IDs
        frame_ids = sorted(frame_detections.keys())
        
        for frame_id in tqdm(frame_ids, desc="Processing detections"):
            detections = frame_detections[frame_id]
            processed_frames += 1
            
            # Track objects
            tracks = self.tracker.update(detections, frame_id)
            
            # Separate forklift and pallet detections
            forklift_tracks = [t for t in tracks if t.class_name == "forklift"]
            pallet_detections = [d for d in detections if d.class_name == "pallet"]
            
            # Spatial analysis (detect pallet carrying)
            spatial_results = self.spatial_analyzer.analyze_frame(
                forklift_tracks, pallet_detections
            )
            
            # Process each forklift
            for track in forklift_tracks:
                tid = track.track_id
                
                # Update pallet carrying status
                if tid in spatial_results:
                    is_carrying, confidence, pallet = spatial_results[tid]
                    track.is_carrying_pallet = is_carrying
                
                # Compute velocity
                velocity = self.motion_estimator.compute_velocity(track)
                track.velocity = velocity
                
                # Classify state
                state = self.state_classifier.classify(
                    track, velocity, track.is_carrying_pallet
                )
                track.update_state(state)
                
                # Store track
                all_tracks[tid] = track
        
        return all_tracks, processed_frames
    
    def _generate_visualization(
        self,
        video_path: Path,
        all_tracks: dict[int, TrackedObject],
        frame_detections: dict[int, list[Detection]],
        frame_interval: int,
        output_name: str
    ) -> None:
        """Generate annotated output video."""
        output_video_path = self.output_dir / "videos" / f"{output_name}_annotated.mp4"
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        reader = VideoReader(video_path)
        writer = VideoWriter(
            output_video_path,
            fps=reader.fps,
            resolution=reader.resolution
        )
        
        # Build frame-to-track mapping for quick lookup
        # Maps roboflow_frame_id -> list of (track_id, detection)
        frame_to_tracks: dict[int, list[tuple[int, Detection, TrackedObject]]] = {}
        for track in all_tracks.values():
            for det in track.detections:
                frame_id = det.frame_id
                if frame_id not in frame_to_tracks:
                    frame_to_tracks[frame_id] = []
                frame_to_tracks[frame_id].append((track.track_id, det, track))
        
        try:
            for video_frame_id, frame in tqdm(reader, desc="Generating video"):
                # Map video frame to Roboflow frame using frame_offset
                # Find the closest Roboflow frame
                roboflow_frame_id = self._get_closest_roboflow_frame(
                    video_frame_id, frame_detections.keys(), frame_interval
                )
                
                # Get detections for this frame
                detections = frame_detections.get(roboflow_frame_id, [])
                
                # Get track info for this frame
                track_info = frame_to_tracks.get(roboflow_frame_id, [])
                
                # Build visualization data
                forklift_detections = [d for d in detections if d.class_name == "forklift"]
                pallet_detections = [d for d in detections if d.class_name == "pallet"]
                
                # Annotate using direct detections (not stored track bboxes)
                annotated = self._annotate_frame_direct(
                    frame, 
                    forklift_detections,
                    pallet_detections,
                    track_info,
                    all_tracks,
                    video_frame_id
                )
                
                writer.write(annotated)
        
        finally:
            reader.release()
            writer.release()
        
        logger.info(f"Annotated video saved: {output_video_path}")
    
    def _get_closest_roboflow_frame(
        self, 
        video_frame_id: int, 
        roboflow_frames, 
        frame_interval: int
    ) -> int:
        """Find the closest Roboflow frame to the video frame."""
        if not roboflow_frames:
            return 0
        
        roboflow_frames_list = sorted(roboflow_frames)
        
        # Find the closest frame in the roboflow results
        # The roboflow_frames are the actual video frame numbers (from frame_offset)
        closest = min(roboflow_frames_list, key=lambda x: abs(x - video_frame_id), default=0)
        return closest
    
    def _annotate_frame_direct(
        self,
        frame: np.ndarray,
        forklift_detections: list[Detection],
        pallet_detections: list[Detection],
        track_info: list[tuple[int, Detection, TrackedObject]],
        all_tracks: dict[int, TrackedObject],
        frame_id: int
    ) -> np.ndarray:
        """Annotate frame using direct detections with track info overlay."""
        import cv2
        
        annotated = frame.copy()
        
        # Color definitions
        COLORS = {
            "forklift": (0, 255, 0),      # Green
            "pallet": (255, 165, 0),       # Orange
            "idle": (0, 0, 255),           # Red
            "moving_empty": (255, 255, 0), # Cyan
            "moving_loaded": (0, 255, 0),  # Green
            "unknown": (128, 128, 128),    # Gray
        }
        
        # Build a map of detection bbox -> track info
        det_to_track = {}
        for track_id, det, track in track_info:
            key = (int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2))
            det_to_track[key] = (track_id, track)
        
        # Draw pallet detections first (background)
        for det in pallet_detections:
            bbox = det.bbox
            pt1 = (int(bbox.x1), int(bbox.y1))
            pt2 = (int(bbox.x2), int(bbox.y2))
            
            cv2.rectangle(annotated, pt1, pt2, COLORS["pallet"], 2)
            
            label = f"pallet {det.confidence:.2f}"
            cv2.putText(annotated, label, (pt1[0], pt1[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["pallet"], 1)
        
        # Draw forklift detections with track info
        for det in forklift_detections:
            bbox = det.bbox
            pt1 = (int(bbox.x1), int(bbox.y1))
            pt2 = (int(bbox.x2), int(bbox.y2))
            
            # Look up track info
            key = (int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2))
            track_id = None
            state = "unknown"
            is_loaded = False
            
            # Find matching track (with some tolerance)
            for tk, (tid, track) in det_to_track.items():
                dist = abs(tk[0] - key[0]) + abs(tk[1] - key[1]) + abs(tk[2] - key[2]) + abs(tk[3] - key[3])
                if dist < 50:  # Tolerance for bbox matching
                    track_id = tid
                    state = track.current_state.value
                    is_loaded = track.is_carrying_pallet
                    break
            
            # Get color based on state
            color = COLORS.get(state, COLORS["unknown"])
            
            # Draw bounding box
            cv2.rectangle(annotated, pt1, pt2, color, 2)
            
            # Build label
            if track_id is not None:
                label = f"#{track_id} {state}"
                if is_loaded:
                    label += " [LOADED]"
            else:
                label = f"forklift {det.confidence:.2f}"
            
            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, 1)
            
            cv2.rectangle(annotated, 
                         (pt1[0], pt1[1] - label_h - 10),
                         (pt1[0] + label_w + 5, pt1[1]),
                         color, -1)
            
            cv2.putText(annotated, label, (pt1[0] + 2, pt1[1] - 5),
                       font, font_scale, (0, 0, 0), 1)
        
        # Draw frame info
        info_text = f"Frame: {frame_id} | Tracks: {len(all_tracks)}"
        cv2.putText(annotated, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated
    
    def _generate_results(
        self,
        tracks: dict[int, TrackedObject],
        total_duration: float,
        output_name: str,
        roboflow_results: dict
    ) -> dict:
        """Generate analytics and reports from processed tracks."""
        # Segment activities
        all_activities = []
        for track in tracks.values():
            activities = self.activity_segmenter.segment(track, fps=self.fps)
            all_activities.extend(activities)
        
        # Generate analytics
        analytics = generate_analytics(
            all_activities,
            total_duration_seconds=total_duration,
            cost_per_hour=self.rules_config.get("analytics", {}).get("cost_per_idle_hour", 75.0),
            forklift_count=len(tracks)
        )
        
        # Generate reports
        self.reporter.generate_json_report(analytics, all_activities, output_name)
        self.reporter.generate_csv_export(all_activities, video_name=output_name)
        self.reporter.save_summary(analytics, video_name=output_name)
        
        # Save raw Roboflow results
        roboflow_output = self.output_dir / "reports" / f"{output_name}_roboflow_raw.json"
        with open(roboflow_output, 'w') as f:
            json.dump(roboflow_results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + generate_summary_report(analytics))
        
        return {
            "analytics": analytics,
            "activities": all_activities,
            "track_count": len(tracks),
            "roboflow_results": roboflow_results
        }
