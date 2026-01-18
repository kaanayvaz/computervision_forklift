"""
Unit tests for ForkliftTracker.

Tests cover:
- Tracker initialization
- Detection to track conversion
- Track persistence across frames
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.entities import Detection, TrackedObject, BoundingBox


class TestTrackerWithMocks:
    """Tests for ForkliftTracker with mocked supervision."""
    
    @pytest.fixture
    def mock_sv_detections(self):
        """Create mock supervision Detections."""
        mock = Mock()
        mock.xyxy = np.array([[100, 200, 300, 400]])
        mock.confidence = np.array([0.85])
        mock.class_id = np.array([0])
        mock.tracker_id = np.array([1])
        return mock
    
    def test_tracker_initialization(self):
        """Test tracker initializes with supervision."""
        with patch("tracking.tracker.SUPERVISION_AVAILABLE", True):
            with patch("tracking.tracker.sv") as mock_sv:
                mock_sv.ByteTrack = Mock()
                
                from tracking.tracker import ForkliftTracker
                
                tracker = ForkliftTracker(
                    track_activation_threshold=0.3,
                    lost_track_buffer=20
                )
                
                mock_sv.ByteTrack.assert_called_once()
    
    def test_tracker_update_empty(self):
        """Test tracker handles empty detections."""
        with patch("tracking.tracker.SUPERVISION_AVAILABLE", True):
            with patch("tracking.tracker.sv") as mock_sv:
                mock_sv.ByteTrack = Mock()
                mock_sv.Detections.empty = Mock(return_value=Mock())
                
                from tracking.tracker import ForkliftTracker
                
                tracker = ForkliftTracker()
                result = tracker.update([], frame_id=0)
                
                assert result == []
    
    def test_detections_to_supervision_format(self):
        """Test conversion of Detection list to supervision format."""
        with patch("tracking.tracker.SUPERVISION_AVAILABLE", True):
            with patch("tracking.tracker.sv") as mock_sv:
                mock_sv.ByteTrack = Mock()
                mock_sv.Detections = Mock()
                
                from tracking.tracker import ForkliftTracker
                
                tracker = ForkliftTracker()
                
                detections = [
                    Detection(
                        bbox=BoundingBox(100, 200, 300, 400),
                        class_id=0,
                        class_name="forklift",
                        confidence=0.9,
                        frame_id=0
                    )
                ]
                
                result = tracker._detections_to_supervision(detections)
                
                # Verify Detections was called with correct arrays
                call_args = mock_sv.Detections.call_args
                assert call_args is not None


class TestTrackedObject:
    """Tests for TrackedObject dataclass."""
    
    def test_tracked_object_creation(self):
        """Test basic TrackedObject creation."""
        obj = TrackedObject(
            track_id=1,
            class_name="forklift"
        )
        
        assert obj.track_id == 1
        assert obj.class_name == "forklift"
        assert obj.detection_count == 0
    
    def test_tracked_object_add_detection(self):
        """Test adding detections to tracked object."""
        obj = TrackedObject(track_id=1, class_name="forklift")
        
        det = Detection(
            bbox=BoundingBox(100, 200, 300, 400),
            class_id=0,
            class_name="forklift",
            confidence=0.9,
            frame_id=0
        )
        
        obj.add_detection(det)
        
        assert obj.detection_count == 1
        assert obj.latest_detection == det
        assert obj.first_frame == 0
        assert obj.last_frame == 0
    
    def test_tracked_object_multiple_detections(self):
        """Test tracked object with multiple detections."""
        obj = TrackedObject(track_id=1, class_name="forklift")
        
        for i in range(5):
            det = Detection(
                bbox=BoundingBox(100 + i*10, 200, 300 + i*10, 400),
                class_id=0,
                class_name="forklift",
                confidence=0.9,
                frame_id=i
            )
            obj.add_detection(det)
        
        assert obj.detection_count == 5
        assert obj.first_frame == 0
        assert obj.last_frame == 4
    
    def test_tracked_object_state_history(self):
        """Test state history management."""
        from core.entities import ForkliftState
        
        obj = TrackedObject(track_id=1, class_name="forklift")
        
        obj.update_state(ForkliftState.IDLE)
        obj.update_state(ForkliftState.MOVING_EMPTY)
        obj.update_state(ForkliftState.MOVING_LOADED)
        
        assert obj.current_state == ForkliftState.MOVING_LOADED
        assert len(obj.state_history) == 3
