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
        confidence: float = 0.25,
        skip_pallet_detection: bool = False  # NEW: Skip pallet detection if causing issues
    ):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.visualize = visualize
        self.fps = fps
        self.confidence = confidence
        self.skip_pallet_detection = skip_pallet_detection  # NEW
        
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
        # Tracking - optimized for forklift tracking with moving cameras
        # Note: CMC disabled because optical flow doesn't work well with sparse frames
        # (Roboflow processes at 3 FPS, so frames are ~8 apart)
        # Instead, we rely on track merging post-processing to combine fragmented tracks
        self.tracker = ForkliftTracker(
            track_activation_threshold=0.3,   # Min confidence to start a track
            lost_track_buffer=500,            # Very high buffer for sparse frames
            minimum_matching_threshold=0.2,   # Very lenient matching (80% overlap needed)
            frame_rate=self.fps,              # Use actual processing FPS
            enable_cmc=False                  # Disabled: doesn't work with sparse frames
        )
        
        # Spatial analysis - use more lenient settings for pallet detection
        # Override config with more relaxed thresholds for better carrying detection
        spatial_config = self.rules_config.get("spatial", {}).copy()
        spatial_config["pallet_iou_threshold"] = spatial_config.get("pallet_iou_threshold", 0.05)  # Lower IoU
        spatial_config["pallet_containment_threshold"] = spatial_config.get("pallet_containment_threshold", 0.20)  # Lower containment
        self.spatial_analyzer = SpatialAnalyzer({"spatial": spatial_config})
        
        # Motion estimation
        motion_config = self.rules_config.get("motion", {})
        self.motion_estimator = MotionEstimator(
            smoothing_window=motion_config.get("smoothing_window", 5)
        )
        
        # State classification - lower idle threshold for sparse frames
        # For sparse frames (3-5 FPS), we need immediate state confirmation
        state_config = self.rules_config.get("state", {})
        self.state_classifier = StateClassifier(
            idle_threshold=motion_config.get("velocity_idle_threshold", 5.0),  # Higher threshold for sparse frames
            confirmation_frames=1  # Immediate confirmation - tracks have few detections
        )
        
        # Activity segmentation - capture all activities regardless of duration
        activity_config = self.rules_config.get("activity", {})
        self.activity_segmenter = ActivitySegmenter(
            min_duration=0.0,  # Capture all activities regardless of duration
            merge_threshold=0.0,  # Don't merge segments
            fps=self.fps  # Use processing FPS for duration calculations
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
            total_frames,
            video_path  # Pass video path for Camera Motion Compensation
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
        
        # Run pallet detection (skip if disabled)
        if self.skip_pallet_detection:
            print("\n📦 Pallet detection: SKIPPED (disabled)")
            results["pallet_detections"] = {}
        else:
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
            confidence = pred.get("confidence", 0)
            
            # Filtering for forklift detections
            # Based on actual Roboflow output analysis:
            # - Confidence: 0.40-0.92, avg 0.74
            # - Width: 98-897, Height: 103-338
            # - Area: 14K-299K
            if class_name == "forklift":
                # 1. Moderate confidence threshold - balance between false positives and detection rate
                min_confidence = 0.65  # Accept 65%+ confidence (filters out obvious false positives)
                if confidence < min_confidence:
                    logger.debug(f"Skipping low-confidence forklift: {confidence:.2f}")
                    return None
                
                # 2. Size constraints based on actual data
                # Real forklifts: ~100-350px in your video
                min_size = 80
                max_size = 400  # Allow larger sizes for closer forklifts
                
                if width < min_size or height < min_size:
                    logger.debug(f"Skipping small forklift detection: {width}x{height}")
                    return None
                
                if width > max_size or height > max_size:
                    logger.debug(f"Skipping oversized forklift detection: {width}x{height}")
                    return None
                
                # 3. Aspect ratio check - forklifts are roughly square-ish (but allow some elongation)
                aspect_ratio = max(width, height) / max(min(width, height), 1)
                if aspect_ratio > 2.5:  # Allow up to 2.5:1 aspect ratio
                    logger.debug(f"Skipping elongated forklift detection: {width}x{height} ratio={aspect_ratio:.1f}")
                    return None
                
                # 4. Area check based on actual data (14K-80K for typical forklifts)
                area = width * height
                if area < 8000 or area > 100000:  # Reasonable forklift area range
                    logger.debug(f"Skipping forklift with unusual area: {area}")
                    return None
                    
            else:  # pallet
                max_size = 400
                min_size = 30
                min_confidence = 0.40
                
                if confidence < min_confidence:
                    return None
                
                if width > max_size or height > max_size:
                    return None
                
                if width < min_size or height < min_size:
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
        total_frames: int,
        video_path: Optional[Path] = None  # NEW: For Camera Motion Compensation
    ) -> tuple[dict[int, TrackedObject], int]:
        """Process frame detections through tracking and state classification.
        
        With Camera Motion Compensation (CMC) enabled, this method reads video frames
        to estimate camera motion and compensate detection positions, improving track
        ID stability when the camera moves.
        """
        all_tracks: dict[int, TrackedObject] = {}
        processed_frames = 0
        
        # HYBRID FILTER 1: Cross-validate forklift/pallet detections before tracking
        # This prevents forklifts from being misclassified as pallets
        frame_detections = self._cross_validate_detections(frame_detections)
        
        # Sort frame IDs
        frame_ids = sorted(frame_detections.keys())
        
        # Note: CMC disabled, so we don't need to read video frames
        
        for frame_id in tqdm(frame_ids, desc="Processing detections"):
            detections = frame_detections[frame_id]
            processed_frames += 1
            
            # IMPORTANT: Only track forklifts, not pallets!
            # Filter detections to only include forklifts for tracking
            forklift_detections = [d for d in detections if d.class_name == "forklift"]
            pallet_detections = [d for d in detections if d.class_name == "pallet"]
            
            # Track only forklift objects
            forklift_tracks = self.tracker.update(forklift_detections, frame_id)
            
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
        
        # TRACK MERGING: Combine fragmented tracks caused by camera motion
        # When camera moves, track IDs can switch even for the same forklift
        # Merge tracks that have similar positions and non-overlapping time ranges
        logger.info(f"Tracks before merging: {len(all_tracks)}")
        all_tracks = self._merge_fragmented_tracks(all_tracks)
        logger.info(f"Tracks after merging: {len(all_tracks)}")
        
        # POST-PROCESSING: Multi-stage hybrid filtering
        logger.info(f"Starting hybrid filtering with {len(all_tracks)} tracks...")
        
        # Filter 1: Remove static objects (false positive forklifts that never move)
        filtered_tracks = self._filter_static_tracks(all_tracks)
        logger.info(f"  After static filter: {len(filtered_tracks)} tracks (removed {len(all_tracks) - len(filtered_tracks)})")
        
        # Filter 2: Remove tracks with inconsistent sizes (flickering false positives)
        prev_count = len(filtered_tracks)
        filtered_tracks = self._filter_inconsistent_tracks(filtered_tracks)
        logger.info(f"  After size consistency filter: {len(filtered_tracks)} tracks (removed {prev_count - len(filtered_tracks)})")
        
        # Filter 3: Remove tracks with erratic motion patterns
        prev_count = len(filtered_tracks)
        filtered_tracks = self._filter_erratic_tracks(filtered_tracks)
        logger.info(f"  After motion pattern filter: {len(filtered_tracks)} tracks (removed {prev_count - len(filtered_tracks)})")
        
        # Filter 4: Remove tracks with fragmented/sparse detections
        prev_count = len(filtered_tracks)
        filtered_tracks = self._filter_fragmented_tracks(filtered_tracks)
        logger.info(f"  After temporal consistency filter: {len(filtered_tracks)} tracks (removed {prev_count - len(filtered_tracks)})")
        
        logger.info(f"Hybrid filtering complete: {len(all_tracks)} -> {len(filtered_tracks)} tracks")
        
        # CRITICAL: Re-classify states AFTER merging and filtering
        # The state_history is lost during merging, so we need to rebuild it
        logger.info("Re-classifying states for merged/filtered tracks...")
        self._classify_track_states(filtered_tracks, frame_detections)
        
        return filtered_tracks, processed_frames
    
    def _classify_track_states(
        self,
        tracks: dict[int, TrackedObject],
        frame_detections: dict[int, list[Detection]]
    ) -> None:
        """
        Re-classify states for all tracks after merging/filtering.
        
        This is CRITICAL because state_history is lost when tracks are merged.
        We rebuild the state history by processing each track's detections in order.
        
        For each detection in the track:
        1. Find pallet detections in the same frame
        2. Determine if forklift is carrying a pallet (spatial analysis)
        3. Compute velocity from detection history
        4. Classify state and add to state_history
        """
        # Reset classifiers for fresh state
        self.motion_estimator.reset()
        self.state_classifier.reset()
        # CRITICAL: Reset size tracking to prevent cross-contamination between merged tracks
        self.spatial_analyzer.reset_size_tracking()
        
        for tid, track in tracks.items():
            # Clear any existing state history
            track.state_history = []
            track.current_state = ForkliftState.UNKNOWN
            
            # Sort detections by frame
            detections = sorted(track.detections, key=lambda d: d.frame_id)
            
            if not detections:
                continue
            
            # Process each detection to build state history
            for i, det in enumerate(detections):
                frame_id = det.frame_id
                
                # Get pallet detections for this frame
                all_frame_dets = frame_detections.get(frame_id, [])
                pallet_dets = [d for d in all_frame_dets if d.class_name == "pallet"]
                
                # Check if carrying pallet using spatial analysis
                # Pass track_id for size-based fallback detection
                is_carrying, confidence, _ = self.spatial_analyzer.is_carrying_pallet(
                    det, pallet_dets, track_id=tid
                )
                track.is_carrying_pallet = is_carrying
                
                # Compute velocity (needs at least 2 detections)
                if i >= 1:
                    prev_det = detections[i - 1]
                    prev_center = prev_det.bbox.center
                    curr_center = det.bbox.center
                    
                    frame_diff = det.frame_id - prev_det.frame_id
                    if frame_diff <= 0:
                        frame_diff = 1
                    
                    displacement = (
                        (curr_center[0] - prev_center[0]) ** 2 +
                        (curr_center[1] - prev_center[1]) ** 2
                    ) ** 0.5
                    velocity = displacement / frame_diff
                else:
                    velocity = 0.0
                
                track.velocity = velocity
                
                # Classify state
                state = self.state_classifier.classify(track, velocity, is_carrying)
                track.update_state(state)
            
            # Log state distribution for this track
            state_counts = {}
            for s in track.state_history:
                state_counts[s.value] = state_counts.get(s.value, 0) + 1
            logger.debug(f"Track #{tid}: {len(track.state_history)} states - {state_counts}")
        
        # Log summary
        total_states = sum(len(t.state_history) for t in tracks.values())
        loaded_count = sum(
            1 for t in tracks.values() 
            for s in t.state_history 
            if s == ForkliftState.MOVING_LOADED
        )
        idle_count = sum(
            1 for t in tracks.values()
            for s in t.state_history
            if s == ForkliftState.IDLE
        )
        moving_empty_count = sum(
            1 for t in tracks.values()
            for s in t.state_history
            if s == ForkliftState.MOVING_EMPTY
        )
        logger.info(f"State classification complete: {total_states} total states")
        logger.info(f"  - MOVING_LOADED: {loaded_count}, MOVING_EMPTY: {moving_empty_count}, IDLE: {idle_count}")
    
    def _merge_fragmented_tracks(self, tracks: dict[int, TrackedObject]) -> dict[int, TrackedObject]:
        """
        Aggressively merge fragmented tracks that likely belong to the same forklift.
        
        At 3 FPS, forklifts can move 300+ pixels between frames, causing ByteTrack to
        lose track IDs. This method uses multiple strategies to consolidate tracks:
        
        1. TEMPORAL MERGING: Sequential tracks with non-overlapping time ranges
        2. VELOCITY PREDICTION: Predict position based on estimated velocity
        3. SPATIAL CLUSTERING: Group tracks by their average position in the scene
        4. MULTI-PASS: Iteratively merge until no more merges possible
        
        The goal is to reduce e.g., 30 fragmented tracks to the actual number of forklifts.
        """
        from core.entities import TrackedObject, Detection
        
        if len(tracks) <= 1:
            return tracks
        
        # Very aggressive configuration for sparse frame tracking
        BASE_POSITION_DISTANCE = 400   # Base distance threshold (pixels)
        MAX_FRAME_GAP = 100            # Allow large gaps (~33 seconds at 3fps)
        MAX_SIZE_RATIO = 2.5           # Allow some size variation
        VELOCITY_SCALE = 1.5           # Scale predicted distance by velocity
        MIN_OVERLAP_IOU = 0.3          # IoU threshold for overlapping tracks
        MAX_MERGE_PASSES = 5           # Maximum merge iterations
        
        def get_track_info(track):
            """Extract comprehensive track metadata."""
            if not track.detections:
                return None
            detections = sorted(track.detections, key=lambda d: d.frame_id)
            first = detections[0]
            last = detections[-1]
            
            first_center = ((first.bbox.x1 + first.bbox.x2) / 2, (first.bbox.y1 + first.bbox.y2) / 2)
            last_center = ((last.bbox.x1 + last.bbox.x2) / 2, (last.bbox.y1 + last.bbox.y2) / 2)
            
            # Calculate average position (centroid of all detections)
            all_centers = [((d.bbox.x1 + d.bbox.x2) / 2, (d.bbox.y1 + d.bbox.y2) / 2) for d in detections]
            avg_center = (sum(c[0] for c in all_centers) / len(all_centers),
                         sum(c[1] for c in all_centers) / len(all_centers))
            
            avg_width = sum(d.bbox.width for d in detections) / len(detections)
            avg_height = sum(d.bbox.height for d in detections) / len(detections)
            
            # Calculate velocity (pixels per frame)
            duration = last.frame_id - first.frame_id
            if duration > 0:
                velocity = (
                    (last_center[0] - first_center[0]) / duration,
                    (last_center[1] - first_center[1]) / duration
                )
            else:
                velocity = (0, 0)
            
            return {
                'start_frame': first.frame_id,
                'end_frame': last.frame_id,
                'first_center': first_center,
                'last_center': last_center,
                'avg_center': avg_center,
                'avg_size': (avg_width, avg_height),
                'velocity': velocity,
                'detections': detections,
                'num_detections': len(detections)
            }
        
        def compute_iou(bbox1, bbox2):
            """Compute IoU between two bboxes."""
            x1 = max(bbox1.x1, bbox2.x1)
            y1 = max(bbox1.y1, bbox2.y1)
            x2 = min(bbox1.x2, bbox2.x2)
            y2 = min(bbox1.y2, bbox2.y2)
            
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            if inter_area == 0:
                return 0.0
            
            area1 = bbox1.width * bbox1.height
            area2 = bbox2.width * bbox2.height
            union_area = area1 + area2 - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
        
        def can_merge_tracks(info1, info2):
            """
            Determine if two tracks can be merged using multiple criteria.
            Returns (can_merge, merge_score) where higher score = better match.
            """
            # Check size similarity first
            size1 = info1['avg_size']
            size2 = info2['avg_size']
            w_ratio = max(size1[0], size2[0]) / max(min(size1[0], size2[0]), 1)
            h_ratio = max(size1[1], size2[1]) / max(min(size1[1], size2[1]), 1)
            
            if w_ratio > MAX_SIZE_RATIO or h_ratio > MAX_SIZE_RATIO:
                return False, 0
            
            # STRATEGY 1: Non-overlapping temporal tracks
            if info1['end_frame'] < info2['start_frame']:
                gap = info2['start_frame'] - info1['end_frame']
                if gap <= MAX_FRAME_GAP:
                    # Use velocity prediction for position matching
                    vel = info1['velocity']
                    predicted_pos = (
                        info1['last_center'][0] + vel[0] * gap,
                        info1['last_center'][1] + vel[1] * gap
                    )
                    actual_pos = info2['first_center']
                    
                    distance = ((predicted_pos[0] - actual_pos[0])**2 + 
                               (predicted_pos[1] - actual_pos[1])**2) ** 0.5
                    
                    # Allow larger distance for longer gaps and faster movement
                    speed = (vel[0]**2 + vel[1]**2) ** 0.5
                    max_dist = BASE_POSITION_DISTANCE + speed * gap * VELOCITY_SCALE
                    
                    if distance < max_dist:
                        # Score based on how close the prediction was
                        score = (max_dist - distance) / max_dist * 100
                        return True, score
                        
            elif info2['end_frame'] < info1['start_frame']:
                # Same logic, reversed
                gap = info1['start_frame'] - info2['end_frame']
                if gap <= MAX_FRAME_GAP:
                    vel = info2['velocity']
                    predicted_pos = (
                        info2['last_center'][0] + vel[0] * gap,
                        info2['last_center'][1] + vel[1] * gap
                    )
                    actual_pos = info1['first_center']
                    
                    distance = ((predicted_pos[0] - actual_pos[0])**2 + 
                               (predicted_pos[1] - actual_pos[1])**2) ** 0.5
                    
                    speed = (vel[0]**2 + vel[1]**2) ** 0.5
                    max_dist = BASE_POSITION_DISTANCE + speed * gap * VELOCITY_SCALE
                    
                    if distance < max_dist:
                        score = (max_dist - distance) / max_dist * 100
                        return True, score
            
            # STRATEGY 2: Overlapping tracks with similar positions (same forklift, ID switch)
            # This catches cases where ByteTrack switches IDs mid-tracking
            overlap_start = max(info1['start_frame'], info2['start_frame'])
            overlap_end = min(info1['end_frame'], info2['end_frame'])
            
            if overlap_start <= overlap_end:
                # Check if the average positions are very close (same forklift)
                avg_dist = ((info1['avg_center'][0] - info2['avg_center'][0])**2 +
                           (info1['avg_center'][1] - info2['avg_center'][1])**2) ** 0.5
                
                if avg_dist < BASE_POSITION_DISTANCE * 0.5:  # Stricter for overlapping
                    # Check IoU of detections in overlapping frames
                    dets1 = {d.frame_id: d for d in info1['detections']}
                    dets2 = {d.frame_id: d for d in info2['detections']}
                    
                    overlap_ious = []
                    for frame_id in range(overlap_start, overlap_end + 1):
                        if frame_id in dets1 and frame_id in dets2:
                            iou = compute_iou(dets1[frame_id].bbox, dets2[frame_id].bbox)
                            overlap_ious.append(iou)
                    
                    if overlap_ious and sum(overlap_ious) / len(overlap_ious) > MIN_OVERLAP_IOU:
                        score = sum(overlap_ious) / len(overlap_ious) * 100
                        return True, score
            
            # STRATEGY 3: Spatial clustering - tracks in same region are likely same forklift
            avg_dist = ((info1['avg_center'][0] - info2['avg_center'][0])**2 +
                       (info1['avg_center'][1] - info2['avg_center'][1])**2) ** 0.5
            
            # Only merge by spatial proximity if tracks don't overlap much in time
            time_overlap = max(0, min(info1['end_frame'], info2['end_frame']) - 
                             max(info1['start_frame'], info2['start_frame']))
            total_time = max(info1['end_frame'], info2['end_frame']) - \
                        min(info1['start_frame'], info2['start_frame'])
            
            overlap_ratio = time_overlap / max(total_time, 1)
            
            if overlap_ratio < 0.2 and avg_dist < BASE_POSITION_DISTANCE * 0.3:
                score = (BASE_POSITION_DISTANCE * 0.3 - avg_dist) / (BASE_POSITION_DISTANCE * 0.3) * 50
                return True, score
            
            return False, 0
        
        # MULTI-PASS MERGING
        current_tracks = tracks.copy()
        
        for pass_num in range(MAX_MERGE_PASSES):
            # Build track info for current tracks
            track_infos = {}
            for tid, track in current_tracks.items():
                info = get_track_info(track)
                if info:
                    track_infos[tid] = info
            
            if len(track_infos) <= 1:
                break
            
            # Find all mergeable pairs with scores
            merge_candidates = []
            track_ids = list(track_infos.keys())
            
            for i, tid1 in enumerate(track_ids):
                for tid2 in track_ids[i+1:]:
                    can_merge, score = can_merge_tracks(track_infos[tid1], track_infos[tid2])
                    if can_merge:
                        merge_candidates.append((tid1, tid2, score))
            
            if not merge_candidates:
                break  # No more merges possible
            
            # Sort by score (highest first) and perform merges
            merge_candidates.sort(key=lambda x: -x[2])
            
            merged = set()
            merge_groups = {}  # tid -> group_id
            next_group = 0
            
            for tid1, tid2, score in merge_candidates:
                if tid1 in merged or tid2 in merged:
                    continue
                
                # Create or extend merge group
                if tid1 in merge_groups:
                    merge_groups[tid2] = merge_groups[tid1]
                elif tid2 in merge_groups:
                    merge_groups[tid1] = merge_groups[tid2]
                else:
                    merge_groups[tid1] = next_group
                    merge_groups[tid2] = next_group
                    next_group += 1
                
                merged.add(tid1)
                merged.add(tid2)
            
            # Build final groups
            groups_by_id = {}
            for tid, group_id in merge_groups.items():
                if group_id not in groups_by_id:
                    groups_by_id[group_id] = []
                groups_by_id[group_id].append(tid)
            
            # Add unmerged tracks as single-item groups
            for tid in track_ids:
                if tid not in merged:
                    groups_by_id[next_group] = [tid]
                    next_group += 1
            
            # Create merged tracks
            new_tracks = {}
            next_track_id = 1
            
            merges_this_pass = 0
            for group in groups_by_id.values():
                all_detections = []
                class_name = "forklift"
                
                for tid in group:
                    all_detections.extend(track_infos[tid]['detections'])
                    class_name = current_tracks[tid].class_name
                
                # Sort by frame and create new track
                all_detections.sort(key=lambda d: d.frame_id)
                new_track = TrackedObject(track_id=next_track_id, class_name=class_name)
                for det in all_detections:
                    new_track.add_detection(det)
                new_tracks[next_track_id] = new_track
                
                if len(group) > 1:
                    merges_this_pass += len(group) - 1
                    logger.debug(f"Pass {pass_num+1}: Merged tracks {group} -> #{next_track_id}")
                
                next_track_id += 1
            
            logger.info(f"Merge pass {pass_num+1}: {len(current_tracks)} -> {len(new_tracks)} tracks ({merges_this_pass} merges)")
            
            if len(new_tracks) == len(current_tracks):
                break  # No change, stop iterating
            
            current_tracks = new_tracks
        
        return current_tracks
    
    def _filter_static_tracks(self, tracks: dict[int, TrackedObject]) -> dict[int, TrackedObject]:
        """
        Filter out tracks that never moved (likely false positives like IBC containers).
        
        A real forklift should show some movement during its tracked lifetime.
        Static objects detected as forklifts are removed.
        """
        filtered = {}
        
        if not tracks:
            logger.warning("No tracks to filter!")
            return filtered
        
        for tid, track in tracks.items():
            # Check if track showed any significant movement
            if len(track.detections) < 2:
                # Single detection - keep it, might be important
                logger.debug(f"Keeping single-detection track #{tid}")
                filtered[tid] = track
                continue
            
            # Calculate total displacement across all detections
            detections = track.detections
            total_displacement = 0.0
            max_displacement = 0.0
            
            first_det = detections[0]
            first_center = ((first_det.bbox.x1 + first_det.bbox.x2) / 2,
                           (first_det.bbox.y1 + first_det.bbox.y2) / 2)
            
            for i in range(1, len(detections)):
                prev = detections[i-1]
                curr = detections[i]
                
                prev_center = ((prev.bbox.x1 + prev.bbox.x2) / 2, (prev.bbox.y1 + prev.bbox.y2) / 2)
                curr_center = ((curr.bbox.x1 + curr.bbox.x2) / 2, (curr.bbox.y1 + curr.bbox.y2) / 2)
                
                displacement = ((curr_center[0] - prev_center[0])**2 + 
                               (curr_center[1] - prev_center[1])**2) ** 0.5
                total_displacement += displacement
                
                # Also track displacement from start
                from_start = ((curr_center[0] - first_center[0])**2 + 
                             (curr_center[1] - first_center[1])**2) ** 0.5
                max_displacement = max(max_displacement, from_start)
            
            # Very relaxed thresholds - we don't want to filter real forklifts
            # Forklifts that appear to be stationary might just be idle
            MIN_TOTAL_MOVEMENT = 5.0   # Very minimal movement threshold
            
            # Always keep tracks with multiple detections - they're likely real
            if len(detections) >= 2:
                filtered[tid] = track
                logger.debug(f"Track #{tid} accepted: {len(detections)} detections, total_disp={total_displacement:.1f}")
            elif total_displacement > MIN_TOTAL_MOVEMENT:
                filtered[tid] = track
                logger.debug(f"Track #{tid} accepted: total_disp={total_displacement:.1f}")
            else:
                logger.debug(f"Filtering static track #{tid}: total_disp={total_displacement:.1f}")
        
        return filtered
    
    def _cross_validate_detections(
        self,
        frame_detections: dict[int, list[Detection]]
    ) -> dict[int, list[Detection]]:
        """
        HYBRID FILTER 1: Cross-validate forklift and pallet detections.
        
        If the same region is detected as both forklift AND pallet:
        - If forklift confidence is much higher (>0.3 difference), keep forklift
        - If pallet confidence is higher, remove the forklift detection
        - This prevents forklifts from being misclassified as pallets
        """
        filtered = {}
        overlap_removals = 0
        
        for frame_id, detections in frame_detections.items():
            forklift_dets = [d for d in detections if d.class_name == "forklift"]
            pallet_dets = [d for d in detections if d.class_name == "pallet"]
            
            # Check each pallet detection for overlap with forklifts
            valid_pallet_dets = []
            for pallet in pallet_dets:
                is_valid_pallet = True
                pallet_center = (
                    (pallet.bbox.x1 + pallet.bbox.x2) / 2,
                    (pallet.bbox.y1 + pallet.bbox.y2) / 2
                )
                
                for forklift in forklift_dets:
                    # Check if pallet center is inside forklift bbox (forklift mistaken as pallet)
                    if (forklift.bbox.x1 <= pallet_center[0] <= forklift.bbox.x2 and
                        forklift.bbox.y1 <= pallet_center[1] <= forklift.bbox.y2):
                        
                        # Calculate overlap ratio
                        overlap = self._calculate_iou(pallet.bbox, forklift.bbox)
                        
                        if overlap > 0.3:  # Significant overlap
                            # Keep forklift, remove pallet (forklift was misdetected as pallet)
                            logger.debug(f"Frame {frame_id}: Removing pallet detection overlapping with forklift (IoU={overlap:.2f})")
                            is_valid_pallet = False
                            overlap_removals += 1
                            break
                
                if is_valid_pallet:
                    valid_pallet_dets.append(pallet)
            
            # Also filter forklifts that completely overlap with high-confidence pallets
            valid_forklift_dets = []
            for forklift in forklift_dets:
                is_valid_forklift = True
                forklift_center = (
                    (forklift.bbox.x1 + forklift.bbox.x2) / 2,
                    (forklift.bbox.y1 + forklift.bbox.y2) / 2
                )
                
                for pallet in pallet_dets:
                    # Only filter if pallet is much more confident
                    if pallet.confidence > forklift.confidence + 0.2:
                        overlap = self._calculate_iou(pallet.bbox, forklift.bbox)
                        if overlap > 0.5:  # High overlap and pallet is more confident
                            logger.debug(f"Frame {frame_id}: Removing weak forklift overlapping with confident pallet")
                            is_valid_forklift = False
                            overlap_removals += 1
                            break
                
                if is_valid_forklift:
                    valid_forklift_dets.append(forklift)
            
            filtered[frame_id] = valid_forklift_dets + valid_pallet_dets
        
        if overlap_removals > 0:
            logger.info(f"Cross-validation removed {overlap_removals} overlapping detections")
        
        return filtered
    
    def _calculate_iou(self, box1: 'BoundingBox', box2: 'BoundingBox') -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _filter_inconsistent_tracks(self, tracks: dict[int, TrackedObject]) -> dict[int, TrackedObject]:
        """
        HYBRID FILTER 2: Filter tracks with inconsistent detection sizes.
        
        Real forklifts maintain relatively consistent size across frames.
        False positives (random objects briefly detected) have wild size variations.
        """
        filtered = {}
        MAX_SIZE_VARIANCE = 0.5  # Maximum 50% size variance (std/mean)
        
        for tid, track in tracks.items():
            if len(track.detections) < 3:
                # Too few detections to judge consistency - already filtered by static filter
                filtered[tid] = track
                continue
            
            # Calculate size statistics
            areas = []
            for det in track.detections:
                width = det.bbox.x2 - det.bbox.x1
                height = det.bbox.y2 - det.bbox.y1
                areas.append(width * height)
            
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            
            if mean_area > 0:
                variance_ratio = std_area / mean_area
            else:
                variance_ratio = 1.0
            
            if variance_ratio > MAX_SIZE_VARIANCE:
                logger.info(f"Filtering inconsistent track #{tid}: size variance={variance_ratio:.2f}")
                continue
            
            filtered[tid] = track
        
        return filtered
    
    def _filter_erratic_tracks(self, tracks: dict[int, TrackedObject]) -> dict[int, TrackedObject]:
        """
        HYBRID FILTER 3: Filter tracks with erratic motion patterns.
        
        Real forklifts move in smooth paths with gradual direction changes.
        False positives (flickering detections) show erratic/jumpy motion.
        """
        filtered = {}
        MAX_DIRECTION_CHANGES = 0.7  # Max 70% of movements can be sharp direction changes
        MIN_DETECTIONS_FOR_MOTION_CHECK = 5
        
        for tid, track in tracks.items():
            if len(track.detections) < MIN_DETECTIONS_FOR_MOTION_CHECK:
                # Too few detections to judge motion pattern
                filtered[tid] = track
                continue
            
            # Calculate direction changes
            detections = track.detections
            direction_changes = 0
            total_moves = 0
            
            prev_dx, prev_dy = None, None
            
            for i in range(1, len(detections)):
                prev = detections[i-1]
                curr = detections[i]
                
                prev_center = ((prev.bbox.x1 + prev.bbox.x2) / 2, (prev.bbox.y1 + prev.bbox.y2) / 2)
                curr_center = ((curr.bbox.x1 + curr.bbox.x2) / 2, (curr.bbox.y1 + curr.bbox.y2) / 2)
                
                dx = curr_center[0] - prev_center[0]
                dy = curr_center[1] - prev_center[1]
                
                # Skip very small movements (noise)
                movement = (dx**2 + dy**2) ** 0.5
                if movement < 3:
                    continue
                
                total_moves += 1
                
                if prev_dx is not None and prev_dy is not None:
                    # Check for sharp direction change (>90 degrees)
                    # Using dot product: negative = opposite direction
                    dot = prev_dx * dx + prev_dy * dy
                    prev_mag = (prev_dx**2 + prev_dy**2) ** 0.5
                    curr_mag = (dx**2 + dy**2) ** 0.5
                    
                    if prev_mag > 0 and curr_mag > 0:
                        cos_angle = dot / (prev_mag * curr_mag)
                        if cos_angle < 0:  # >90 degree change
                            direction_changes += 1
                
                prev_dx, prev_dy = dx, dy
            
            # Calculate erratic ratio
            if total_moves > 2:
                erratic_ratio = direction_changes / total_moves
                if erratic_ratio > MAX_DIRECTION_CHANGES:
                    logger.info(f"Filtering erratic track #{tid}: direction_changes={erratic_ratio:.2f}")
                    continue
            
            filtered[tid] = track
        
        return filtered
    
    def _filter_fragmented_tracks(self, tracks: dict[int, TrackedObject]) -> dict[int, TrackedObject]:
        """
        HYBRID FILTER 4: Filter tracks with fragmented/sparse detections.
        
        Real forklifts should be detected in multiple frames, but at 3 FPS with
        cloud detection, we may only get 3-15 detections per forklift in a 39-second video.
        
        This filter now only removes tracks that are OBVIOUSLY noise:
        - Single detection tracks with huge span
        - Extremely scattered detections (1-2 detections across 100+ frames)
        """
        filtered = {}
        
        # Very relaxed: at 3 FPS for 39 seconds, we have ~117 frames
        # A forklift visible for 50% of the video at 3 FPS would have ~58 frames
        # But detection might miss many, so accept as low as 3%
        MIN_DENSITY = 0.02  # Only 2% density required
        MIN_DETECTIONS = 2  # Need at least 2 detections to not be noise
        MIN_SPAN_FOR_FILTER = 50  # Only apply density filter for long spans
        
        for tid, track in tracks.items():
            num_detections = len(track.detections)
            
            # Keep all tracks with multiple detections
            if num_detections >= MIN_DETECTIONS:
                # Get frame IDs
                frame_ids = sorted([det.frame_id for det in track.detections])
                span = frame_ids[-1] - frame_ids[0] + 1
                
                # Only check density for tracks with very long spans
                if span > MIN_SPAN_FOR_FILTER:
                    density = num_detections / span
                    if density < MIN_DENSITY:
                        logger.info(f"Filtering sparse track #{tid}: {num_detections} detections over {span} frames (density={density:.3f})")
                        continue
                
                filtered[tid] = track
                logger.debug(f"Keeping track #{tid}: {num_detections} detections over span={span if num_detections > 1 else 0}")
            else:
                logger.debug(f"Filtering single-detection track #{tid}")
        
        return filtered
    
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
