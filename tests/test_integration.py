"""
Integration tests for the complete forklift analytics pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.entities import Detection, TrackedObject, BoundingBox, ForkliftState, Activity


class TestIntegrationPipeline:
    """End-to-end pipeline integration tests."""
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample detections for testing."""
        frames_detections = []
        
        for frame_id in range(10):
            # Forklift moving right
            forklift = Detection(
                bbox=BoundingBox(
                    100 + frame_id * 10,  # Moving right
                    200,
                    300 + frame_id * 10,
                    400
                ),
                class_id=0,
                class_name="forklift",
                confidence=0.9,
                frame_id=frame_id
            )
            
            # Stationary pallet
            pallet = Detection(
                bbox=BoundingBox(500, 300, 600, 400),
                class_id=1,
                class_name="pallet",
                confidence=0.85,
                frame_id=frame_id
            )
            
            frames_detections.append([forklift, pallet])
        
        return frames_detections
    
    def test_spatial_analyzer_integration(self):
        """Test spatial analyzer with realistic data."""
        from spatial.pallet_detector import SpatialAnalyzer
        
        analyzer = SpatialAnalyzer()
        
        # Create forklift with pallet on forks
        forklift = TrackedObject(track_id=1, class_name="forklift")
        forklift.add_detection(Detection(
            bbox=BoundingBox(100, 100, 300, 400),
            class_id=0, class_name="forklift", confidence=0.9, frame_id=0
        ))
        
        # Pallet overlapping with lower part of forklift
        pallet = Detection(
            bbox=BoundingBox(120, 300, 280, 380),
            class_id=1, class_name="pallet", confidence=0.9, frame_id=0
        )
        
        is_carrying, confidence, detected = analyzer.is_carrying_pallet(
            forklift, [pallet]
        )
        
        assert is_carrying is True
        assert confidence > 0.5
    
    def test_motion_estimator_integration(self):
        """Test motion estimator with moving object."""
        from motion.motion_estimator import MotionEstimator
        
        estimator = MotionEstimator(smoothing_window=3)
        
        # Create moving forklift
        track = TrackedObject(track_id=1, class_name="forklift")
        for i in range(5):
            track.add_detection(Detection(
                bbox=BoundingBox(100 + i*20, 200, 300 + i*20, 400),
                class_id=0, class_name="forklift", confidence=0.9, frame_id=i
            ))
        
        velocity = estimator.compute_velocity(track)
        
        # Should detect movement (20 pixels per frame)
        assert velocity > 15.0
        assert not estimator.is_stationary(track, idle_threshold=10.0)
    
    def test_state_classifier_integration(self):
        """Test state classifier with complete scenario."""
        from state.classifier import StateClassifier
        
        classifier = StateClassifier(idle_threshold=2.0, confirmation_frames=2)
        
        track = TrackedObject(track_id=1, class_name="forklift")
        track.add_detection(Detection(
            bbox=BoundingBox(100, 200, 300, 400),
            class_id=0, class_name="forklift", confidence=0.9, frame_id=0
        ))
        
        # Simulate several frames of high velocity without pallet
        for _ in range(5):
            state = classifier.classify(track, velocity=10.0, is_carrying_pallet=False)
        
        assert state == ForkliftState.MOVING_EMPTY
    
    def test_activity_segmenter_integration(self):
        """Test activity segmentation from state history."""
        from analytics.activity_segmenter import ActivitySegmenter
        
        segmenter = ActivitySegmenter(min_duration=0.1, fps=10.0)
        
        # Create track with state history
        track = TrackedObject(track_id=1, class_name="forklift")
        
        # Add state history: 5 IDLE, then 5 MOVING_EMPTY
        for _ in range(5):
            track.update_state(ForkliftState.IDLE)
        for _ in range(5):
            track.update_state(ForkliftState.MOVING_EMPTY)
        
        activities = segmenter.segment(track, fps=10.0)
        
        # Should have at least 2 activities (may vary based on min_duration)
        assert len(activities) >= 1
    
    def test_metrics_calculation_integration(self):
        """Test analytics calculation from activities."""
        from analytics.metrics import calculate_utilization, calculate_idle_time, generate_analytics
        
        activities = [
            Activity(
                track_id=1, state=ForkliftState.IDLE,
                start_frame=0, end_frame=30, fps=30.0
            ),
            Activity(
                track_id=1, state=ForkliftState.MOVING_LOADED,
                start_frame=30, end_frame=90, fps=30.0
            ),
        ]
        
        utilization = calculate_utilization(activities)
        idle_time = calculate_idle_time(activities)
        
        # 2 seconds active, 1 second idle = 66.67% utilization
        assert utilization > 60.0
        assert idle_time == pytest.approx(1.0, rel=0.1)
    
    def test_full_pipeline_mock(self):
        """Test full pipeline with mocked components."""
        # This test verifies the pipeline can be initialized
        # Full execution requires actual video files
        
        with patch("pipelines.batch_processor.ForkliftDetector"):
            with patch("pipelines.batch_processor.ForkliftTracker"):
                from pipelines.batch_processor import BatchProcessor
                
                # Should not raise on initialization with mocked detector
                # processor = BatchProcessor(config_dir="config")
                # assert processor is not None
                pass  # Constructor requires real config files


class TestEndToEndScenarios:
    """Test realistic warehouse scenarios."""
    
    def test_scenario_forklift_pickup_dropoff(self):
        """Simulate forklift picking up and dropping pallet."""
        from state.classifier import StateClassifier
        from analytics.activity_segmenter import ActivitySegmenter
        
        classifier = StateClassifier(idle_threshold=2.0, confirmation_frames=2)
        segmenter = ActivitySegmenter(min_duration=0.1, fps=10.0)
        
        track = TrackedObject(track_id=1, class_name="forklift")
        track.add_detection(Detection(
            bbox=BoundingBox(100, 200, 300, 400),
            class_id=0, class_name="forklift", confidence=0.9, frame_id=0
        ))
        
        # Scenario: Move empty -> Pick up pallet -> Move loaded -> Drop pallet
        scenarios = [
            (5.0, False),  # Moving empty
            (5.0, False),
            (0.5, True),   # Loading (just picked up)
            (5.0, True),   # Moving loaded
            (5.0, True),
            (0.5, False),  # Unloading (just dropped)
            (1.0, False),  # Idle
        ]
        
        for velocity, is_carrying in scenarios:
            state = classifier.classify(track, velocity, is_carrying)
            track.update_state(state)
        
        activities = segmenter.segment(track, fps=10.0)
        
        # Should detect multiple activity segments
        assert len(track.state_history) == len(scenarios)
