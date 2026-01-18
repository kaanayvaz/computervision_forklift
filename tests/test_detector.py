"""
Unit tests for ForkliftDetector.

Tests cover:
- Model initialization
- Detection result parsing
- Class filtering
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.entities import Detection, BoundingBox


class TestDetection:
    """Tests for Detection dataclass."""
    
    def test_detection_creation(self):
        """Test basic detection creation."""
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        det = Detection(
            bbox=bbox,
            class_id=0,
            class_name="forklift",
            confidence=0.85,
            frame_id=0
        )
        
        assert det.class_name == "forklift"
        assert det.confidence == 0.85
        assert det.frame_id == 0
    
    def test_detection_invalid_confidence(self):
        """Test detection rejects invalid confidence."""
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        
        with pytest.raises(ValueError):
            Detection(
                bbox=bbox,
                class_id=0,
                class_name="forklift",
                confidence=1.5,  # Invalid
                frame_id=0
            )
    
    def test_detection_invalid_frame_id(self):
        """Test detection rejects negative frame ID."""
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        
        with pytest.raises(ValueError):
            Detection(
                bbox=bbox,
                class_id=0,
                class_name="forklift",
                confidence=0.5,
                frame_id=-1  # Invalid
            )


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""
    
    def test_bbox_properties(self):
        """Test bounding box computed properties."""
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        
        assert bbox.width == 200
        assert bbox.height == 200
        assert bbox.area == 40000
        assert bbox.center == (200, 300)
    
    def test_bbox_auto_correct(self):
        """Test bbox auto-corrects swapped coordinates."""
        bbox = BoundingBox(x1=300, y1=400, x2=100, y2=200)
        
        assert bbox.x1 == 100
        assert bbox.y1 == 200
        assert bbox.x2 == 300
        assert bbox.y2 == 400
    
    def test_bbox_as_tuple(self):
        """Test bbox tuple conversion."""
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        
        assert bbox.as_tuple() == (100, 200, 300, 400)
    
    def test_bbox_as_xywh(self):
        """Test bbox xywh conversion."""
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        
        assert bbox.as_xywh() == (100, 200, 200, 200)


class TestForkliftDetectorMocked:
    """Tests for ForkliftDetector with mocked YOLO."""
    
    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create mock config file."""
        config_content = """
model:
  weights_path: "test.pt"
  device: "cpu"
  half_precision: false

detection:
  confidence_threshold: 0.5
  iou_threshold: 0.45
  classes:
    forklift: 0
    pallet: 1
    person: 2
"""
        config_file = tmp_path / "inference.yaml"
        config_file.write_text(config_content)
        return config_file
    
    @pytest.fixture
    def mock_yolo_result(self):
        """Create mock YOLO result."""
        import torch
        
        mock_boxes = Mock()
        mock_boxes.xyxy = torch.tensor([[100.0, 200.0, 300.0, 400.0]])
        mock_boxes.conf = torch.tensor([0.85])
        mock_boxes.cls = torch.tensor([0])
        mock_boxes.__len__ = Mock(return_value=1)
        
        mock_result = Mock()
        mock_result.boxes = mock_boxes
        
        return mock_result
    
    def test_detector_parse_results(self, mock_config, mock_yolo_result):
        """Test parsing YOLO results to Detection objects."""
        with patch("detection.detector._get_yolo") as mock_get_yolo:
            mock_model = Mock()
            mock_get_yolo.return_value = Mock(return_value=mock_model)
            
            from detection.detector import ForkliftDetector
            
            detector = ForkliftDetector(mock_config)
            detections = detector._parse_single_result(mock_yolo_result, frame_id=5)
            
            assert len(detections) == 1
            assert detections[0].class_name == "forklift"
            assert detections[0].confidence == pytest.approx(0.85, rel=0.01)
            assert detections[0].frame_id == 5
    
    def test_filter_by_class(self, mock_config):
        """Test filtering detections by class."""
        with patch("detection.detector._get_yolo") as mock_get_yolo:
            mock_model = Mock()
            mock_get_yolo.return_value = Mock(return_value=mock_model)
            
            from detection.detector import ForkliftDetector
            
            detector = ForkliftDetector(mock_config)
            
            # Create test detections
            detections = [
                Detection(
                    bbox=BoundingBox(0, 0, 100, 100),
                    class_id=0,
                    class_name="forklift",
                    confidence=0.9,
                    frame_id=0
                ),
                Detection(
                    bbox=BoundingBox(200, 200, 300, 300),
                    class_id=1,
                    class_name="pallet",
                    confidence=0.8,
                    frame_id=0
                ),
            ]
            
            forklifts = detector.get_forklifts(detections)
            pallets = detector.get_pallets(detections)
            
            assert len(forklifts) == 1
            assert forklifts[0].class_name == "forklift"
            
            assert len(pallets) == 1
            assert pallets[0].class_name == "pallet"
