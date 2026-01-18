"""
Unit tests for SpatialAnalyzer (pallet-on-forklift detection).
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.entities import Detection, TrackedObject, BoundingBox


class TestSpatialAnalyzer:
    """Tests for SpatialAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create SpatialAnalyzer with default config."""
        from spatial.pallet_detector import SpatialAnalyzer
        return SpatialAnalyzer({
            "spatial": {
                "pallet_iou_threshold": 0.3,
                "pallet_containment_threshold": 0.5,
                "vertical_offset_max": 50,
                "fork_zone_ratio": 0.4
            }
        })
    
    @pytest.fixture
    def forklift_bbox(self):
        """Create a sample forklift bounding box."""
        return BoundingBox(x1=100, y1=100, x2=300, y2=400)
    
    def test_no_pallets_returns_false(self, analyzer, forklift_bbox):
        """Test that no pallets means not carrying."""
        is_carrying, confidence, pallet = analyzer.is_carrying_pallet(
            forklift_bbox, []
        )
        
        assert is_carrying is False
        assert confidence == 0.0
        assert pallet is None
    
    def test_pallet_on_forks_detected(self, analyzer, forklift_bbox):
        """Test pallet in fork zone is detected as carrying."""
        # Create pallet overlapping with lower portion of forklift
        pallet = Detection(
            bbox=BoundingBox(x1=120, y1=300, x2=280, y2=380),
            class_id=1,
            class_name="pallet",
            confidence=0.9,
            frame_id=0
        )
        
        is_carrying, confidence, detected_pallet = analyzer.is_carrying_pallet(
            forklift_bbox, [pallet]
        )
        
        assert is_carrying is True
        assert confidence > 0.5
        assert detected_pallet == pallet
    
    def test_ground_pallet_not_detected(self, analyzer, forklift_bbox):
        """Test pallet on ground (below forklift) is not carrying."""
        # Create pallet below forklift
        pallet = Detection(
            bbox=BoundingBox(x1=100, y1=450, x2=300, y2=550),
            class_id=1,
            class_name="pallet",
            confidence=0.9,
            frame_id=0
        )
        
        is_carrying, confidence, detected_pallet = analyzer.is_carrying_pallet(
            forklift_bbox, [pallet]
        )
        
        assert is_carrying is False
    
    def test_distant_pallet_not_detected(self, analyzer, forklift_bbox):
        """Test distant pallet is not detected as carrying."""
        # Create pallet far from forklift
        pallet = Detection(
            bbox=BoundingBox(x1=500, y1=500, x2=600, y2=600),
            class_id=1,
            class_name="pallet",
            confidence=0.9,
            frame_id=0
        )
        
        is_carrying, confidence, detected_pallet = analyzer.is_carrying_pallet(
            forklift_bbox, [pallet]
        )
        
        assert is_carrying is False
    
    def test_relative_position_on_forks(self, analyzer):
        """Test relative position returns 'on_forks' for overlapping."""
        forklift = BoundingBox(x1=100, y1=100, x2=300, y2=400)
        pallet = BoundingBox(x1=120, y1=300, x2=280, y2=380)
        
        position = analyzer.get_relative_position(forklift, pallet)
        
        assert position == "on_forks"
    
    def test_relative_position_below(self, analyzer):
        """Test relative position returns 'below' for ground pallet."""
        forklift = BoundingBox(x1=100, y1=100, x2=300, y2=400)
        pallet = BoundingBox(x1=100, y1=450, x2=300, y2=550)
        
        position = analyzer.get_relative_position(forklift, pallet)
        
        assert position == "below"
    
    def test_analyze_frame_multiple_forklifts(self, analyzer):
        """Test analyzing multiple forklifts in a frame."""
        # Create two forklift tracks
        track1 = TrackedObject(track_id=1, class_name="forklift")
        track1.add_detection(Detection(
            bbox=BoundingBox(100, 100, 300, 400),
            class_id=0, class_name="forklift", confidence=0.9, frame_id=0
        ))
        
        track2 = TrackedObject(track_id=2, class_name="forklift")
        track2.add_detection(Detection(
            bbox=BoundingBox(500, 100, 700, 400),
            class_id=0, class_name="forklift", confidence=0.9, frame_id=0
        ))
        
        # Create one pallet on forklift 1
        pallet = Detection(
            bbox=BoundingBox(120, 300, 280, 380),
            class_id=1, class_name="pallet", confidence=0.9, frame_id=0
        )
        
        results = analyzer.analyze_frame([track1, track2], [pallet])
        
        assert results[1][0] is True  # Track 1 carrying
        assert results[2][0] is False  # Track 2 not carrying
