"""
Pytest configuration and fixtures for forklift analytics tests.
"""

import pytest
import numpy as np
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_frame():
    """Create a sample BGR frame for testing."""
    # 720p frame with random noise
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox():
    """Create a sample bounding box."""
    from core.entities import BoundingBox
    return BoundingBox(x1=100, y1=200, x2=300, y2=400)


@pytest.fixture
def sample_detection(sample_bbox):
    """Create a sample detection."""
    from core.entities import Detection
    return Detection(
        bbox=sample_bbox,
        class_id=0,
        class_name="forklift",
        confidence=0.85,
        frame_id=0
    )


@pytest.fixture
def sample_tracked_object(sample_detection):
    """Create a sample tracked object with some detections."""
    from core.entities import TrackedObject, ForkliftState, BoundingBox, Detection
    
    tracked = TrackedObject(
        track_id=1,
        class_name="forklift"
    )
    
    # Add several detections at different positions
    for i in range(5):
        det = Detection(
            bbox=BoundingBox(
                x1=100 + i * 10,
                y1=200,
                x2=300 + i * 10,
                y2=400
            ),
            class_id=0,
            class_name="forklift",
            confidence=0.85,
            frame_id=i
        )
        tracked.add_detection(det)
    
    return tracked


@pytest.fixture
def config_dir(tmp_path):
    """Create temporary config directory with sample configs."""
    config_path = tmp_path / "config"
    config_path.mkdir()
    
    # Create inference.yaml
    inference_config = """
model:
  weights_path: "models/yolov8s.pt"
  device: "cpu"
  half_precision: false

detection:
  confidence_threshold: 0.5
  iou_threshold: 0.45
  classes:
    forklift: 0
    pallet: 1
    person: 2

processing:
  batch_size: 1
  frame_skip: 1
  max_resolution: [1280, 720]
"""
    (config_path / "inference.yaml").write_text(inference_config)
    
    # Create rules.yaml
    rules_config = """
spatial:
  pallet_iou_threshold: 0.3
  pallet_containment_threshold: 0.5
  vertical_offset_max: 50

motion:
  velocity_idle_threshold: 2.0
  smoothing_window: 5

state:
  idle_duration_threshold: 30
  operator_absent_timeout: 60
"""
    (config_path / "rules.yaml").write_text(rules_config)
    
    return config_path


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent
