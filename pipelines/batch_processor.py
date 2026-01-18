"""
Batch processing pipeline for video analysis.

Orchestrates the full processing flow:
1. Video reading
2. Detection
3. Tracking
4. Spatial analysis
5. Motion estimation
6. State classification
7. Activity segmentation
8. Analytics generation
9. Report output
"""

from pathlib import Path
from typing import Optional
from tqdm import tqdm

from core.entities import TrackedObject, ForkliftState
from core.utils import get_logger, load_config, FPSTracker
from video.reader import VideoReader, VideoWriter
from detection.detector import ForkliftDetector
from tracking.tracker import ForkliftTracker
from spatial.pallet_detector import SpatialAnalyzer
from motion.motion_estimator import MotionEstimator
from state.classifier import StateClassifier
from analytics.activity_segmenter import ActivitySegmenter
from analytics.metrics import generate_analytics, generate_summary_report
from analytics.reporter import Reporter
from visualization.visualizer import Visualizer

logger = get_logger(__name__)


class BatchProcessor:
    """
    End-to-end batch processing pipeline for forklift video analysis.
    
    Processes offline video files through the complete detection,
    tracking, classification, and analytics pipeline.
    
    Args:
        config_dir: Path to configuration directory.
        output_dir: Path for output files.
        visualize: Whether to generate annotated output video.
    """
    
    def __init__(
        self,
        config_dir: str | Path = "config",
        output_dir: str | Path = "data/outputs",
        visualize: bool = True
    ):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.visualize = visualize
        
        # Load configurations
        self.inference_config = load_config(self.config_dir / "inference.yaml")
        self.rules_config = load_config(self.config_dir / "rules.yaml")
        
        # Initialize components
        self._init_components()
        
        logger.info(f"BatchProcessor initialized: config={config_dir}, output={output_dir}")
    
    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        # Detection
        self.detector = ForkliftDetector(self.config_dir / "inference.yaml")
        
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
        Process a video file through the complete pipeline.
        
        Args:
            video_path: Path to input video.
            output_name: Optional name for output files.
            
        Returns:
            Dictionary with processing results and analytics.
        """
        video_path = Path(video_path)
        if output_name is None:
            output_name = video_path.stem
        
        logger.info(f"Processing video: {video_path}")
        
        # Reset components
        self.tracker.reset()
        self.motion_estimator.reset()
        self.state_classifier.reset()
        
        # Open video
        processing_config = self.inference_config.get("processing", {})
        reader = VideoReader(
            video_path,
            frame_skip=processing_config.get("frame_skip", 1),
            max_resolution=tuple(processing_config.get("max_resolution", [1280, 720]))
        )
        
        # Setup output video writer
        video_writer = None
        if self.visualize:
            output_video_path = self.output_dir / "videos" / f"{output_name}_annotated.mp4"
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            video_writer = VideoWriter(
                output_video_path,
                fps=reader.effective_fps,
                resolution=reader.resolution
            )
        
        # Process frames
        all_tracks: dict[int, TrackedObject] = {}
        
        try:
            for frame_id, frame in tqdm(reader, desc="Processing frames"):
                self.fps_tracker.tick()
                
                # Detect objects
                detections = self.detector.detect_frame(frame, frame_id)
                
                # Track objects
                tracks = self.tracker.update(detections, frame_id)
                
                # Get forklift and pallet detections
                forklift_tracks = [t for t in tracks if t.class_name == "forklift"]
                pallet_detections = self.detector.get_pallets(detections)
                
                # Spatial analysis (pallet carrying)
                spatial_results = self.spatial_analyzer.analyze_frame(
                    forklift_tracks, pallet_detections
                )
                
                # Update carrying status and classify state for each forklift
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
                
                # Visualize if enabled
                if video_writer:
                    annotated = self.visualizer.annotate_frame(
                        frame, forklift_tracks, pallet_detections, frame_id
                    )
                    video_writer.write(annotated)
        
        finally:
            reader.release()
            if video_writer:
                video_writer.release()
        
        # Generate analytics
        results = self._generate_results(
            all_tracks, reader.duration_seconds, output_name
        )
        
        logger.info(f"Processing complete: {len(all_tracks)} forklifts tracked")
        logger.info(f"Average FPS: {self.fps_tracker.fps:.1f}")
        
        return results
    
    def _generate_results(
        self,
        tracks: dict[int, TrackedObject],
        total_duration: float,
        output_name: str
    ) -> dict:
        """Generate analytics and reports from processed tracks."""
        # Segment activities
        all_activities = []
        for track in tracks.values():
            activities = self.activity_segmenter.segment(track)
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
        
        # Print summary
        print(generate_summary_report(analytics))
        
        return {
            "analytics": analytics,
            "activities": all_activities,
            "track_count": len(tracks),
            "avg_fps": self.fps_tracker.fps
        }
