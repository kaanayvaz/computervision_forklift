"""
Unit tests for StateClassifier and rules.
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.entities import TrackedObject, Detection, BoundingBox, ForkliftState


class TestStateClassifier:
    """Tests for StateClassifier class."""
    
    @pytest.fixture
    def classifier(self):
        """Create StateClassifier with default config."""
        from state.classifier import StateClassifier
        return StateClassifier(
            idle_threshold=2.0,
            confirmation_frames=3
        )
    
    @pytest.fixture
    def sample_track(self):
        """Create sample tracked forklift."""
        track = TrackedObject(track_id=1, class_name="forklift")
        # Add some detection history
        for i in range(5):
            det = Detection(
                bbox=BoundingBox(100 + i, 200, 300 + i, 400),
                class_id=0,
                class_name="forklift",
                confidence=0.9,
                frame_id=i
            )
            track.add_detection(det)
        return track
    
    def test_idle_classification(self, classifier, sample_track):
        """Test forklift classified as IDLE when stationary without pallet."""
        # Call classify multiple times to get past confirmation threshold
        for _ in range(5):
            state = classifier.classify(
                sample_track,
                velocity=1.0,  # Below threshold
                is_carrying_pallet=False
            )
        
        assert state == ForkliftState.IDLE
    
    def test_moving_empty_classification(self, classifier, sample_track):
        """Test forklift classified as MOVING_EMPTY when moving without pallet."""
        for _ in range(5):
            state = classifier.classify(
                sample_track,
                velocity=5.0,  # Above threshold
                is_carrying_pallet=False
            )
        
        assert state == ForkliftState.MOVING_EMPTY
    
    def test_moving_loaded_classification(self, classifier, sample_track):
        """Test forklift classified as MOVING_LOADED when moving with pallet."""
        for _ in range(5):
            state = classifier.classify(
                sample_track,
                velocity=5.0,  # Above threshold
                is_carrying_pallet=True
            )
        
        assert state == ForkliftState.MOVING_LOADED
    
    def test_state_confirmation_prevents_flicker(self, classifier, sample_track):
        """Test that state changes require confirmation frames."""
        # Start with IDLE
        for _ in range(5):
            classifier.classify(sample_track, velocity=1.0, is_carrying_pallet=False)
        
        # Single high velocity should not immediately change state
        state = classifier.classify(sample_track, velocity=10.0, is_carrying_pallet=False)
        
        # Should still be IDLE (not enough confirmation)
        assert state == ForkliftState.IDLE
    
    def test_classify_batch(self, classifier):
        """Test batch classification of multiple tracks."""
        tracks = []
        for i in range(3):
            track = TrackedObject(track_id=i, class_name="forklift")
            track.add_detection(Detection(
                bbox=BoundingBox(100, 200, 300, 400),
                class_id=0, class_name="forklift", confidence=0.9, frame_id=0
            ))
            tracks.append(track)
        
        velocities = {0: 1.0, 1: 5.0, 2: 5.0}
        pallet_status = {0: False, 1: False, 2: True}
        
        # Classify multiple times to get past confirmation
        for _ in range(5):
            results = classifier.classify_batch(tracks, velocities, pallet_status)
        
        assert results[0] == ForkliftState.IDLE
        assert results[1] == ForkliftState.MOVING_EMPTY
        assert results[2] == ForkliftState.MOVING_LOADED


class TestMotionEstimator:
    """Tests for MotionEstimator class."""
    
    @pytest.fixture
    def estimator(self):
        """Create MotionEstimator with default config."""
        from motion.motion_estimator import MotionEstimator
        return MotionEstimator(smoothing_window=3, min_history_length=2)
    
    @pytest.fixture
    def moving_track(self):
        """Create tracked object with motion."""
        track = TrackedObject(track_id=1, class_name="forklift")
        # Add detections with consistent movement
        for i in range(5):
            det = Detection(
                bbox=BoundingBox(100 + i*20, 200, 300 + i*20, 400),
                class_id=0,
                class_name="forklift",
                confidence=0.9,
                frame_id=i
            )
            track.add_detection(det)
        return track
    
    @pytest.fixture
    def stationary_track(self):
        """Create tracked object without motion."""
        track = TrackedObject(track_id=2, class_name="forklift")
        # Add detections at same position
        for i in range(5):
            det = Detection(
                bbox=BoundingBox(100, 200, 300, 400),
                class_id=0,
                class_name="forklift",
                confidence=0.9,
                frame_id=i
            )
            track.add_detection(det)
        return track
    
    def test_velocity_calculation(self, estimator, moving_track):
        """Test velocity is calculated correctly for moving object."""
        velocity = estimator.compute_velocity(moving_track)
        
        # Expected: 20 pixels per frame movement
        assert velocity > 15.0  # Allow for smoothing effects
    
    def test_stationary_detection(self, estimator, stationary_track):
        """Test stationary object has near-zero velocity."""
        velocity = estimator.compute_velocity(stationary_track)
        
        assert velocity < 1.0
    
    def test_is_stationary(self, estimator, moving_track, stationary_track):
        """Test is_stationary method."""
        assert estimator.is_stationary(stationary_track, idle_threshold=2.0) is True
        assert estimator.is_stationary(moving_track, idle_threshold=2.0) is False


class TestStateRules:
    """Tests for individual state rules."""
    
    def test_apply_rules_idle(self):
        """Test rules correctly identify idle state."""
        from state.rules import apply_rules
        
        state, reason = apply_rules(
            velocity=1.0,
            is_carrying=False,
            was_carrying=False,
            threshold=2.0
        )
        
        assert state == ForkliftState.IDLE
        assert "Stationary" in reason
    
    def test_apply_rules_moving_loaded(self):
        """Test rules correctly identify moving loaded state."""
        from state.rules import apply_rules
        
        state, reason = apply_rules(
            velocity=5.0,
            is_carrying=True,
            was_carrying=True,
            threshold=2.0
        )
        
        assert state == ForkliftState.MOVING_LOADED
        assert "Transporting" in reason
    
    def test_apply_rules_loading(self):
        """Test rules correctly identify loading state."""
        from state.rules import apply_rules
        
        state, reason = apply_rules(
            velocity=0.5,
            is_carrying=True,
            was_carrying=False,
            threshold=2.0
        )
        
        assert state == ForkliftState.LOADING
