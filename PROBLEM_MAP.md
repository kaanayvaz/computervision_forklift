# 🗺️ Problem Map & Solutions Guide

## Overview

This document maps all detected problems in the Forklift Analytics codebase and provides detailed solutions for each issue.

---

## ✅ VERIFIED WORKING (No Issues Found)

The following initially suspected issues were **verified to be working correctly**:

| Component | Method | Status |
|-----------|--------|--------|
| MotionEstimator | `reset()` | ✅ Exists (line 179) |
| StateClassifier | `reset()` | ✅ Exists (line 243) |
| SpatialAnalyzer | `analyze_frame()` | ✅ Exists (line 250+) |
| ForkliftTracker | `reset()` | ✅ Exists (line 268) |

---

## 🔴 CRITICAL PROBLEMS (Must Fix)

### Problem #1: Pallet Detection Model Mismatch

**Location:** `config/inference.yaml`

**Issue:** The inference config specifies:
```yaml
classes:
  forklift: 0
  person: 1
```

But the system expects **pallet** detection for the `SpatialAnalyzer` to work. Without pallet detections, `is_carrying_pallet()` will always return `False`.

**Impact:** Pallet carrying status will never be detected.

**Solutions (choose one):**

**Option A: Add pallet detection model**
```yaml
# Use a model trained on pallets OR use two models
detection:
  classes:
    forklift: 0
    pallet: 1  
    person: 2
```

**Option B: Use Roboflow pallet detection**
Use the existing dataset in `data/datasets/pallet/` to train a pallet detection model, or use the `process_video_roboflow.py` script which may support Roboflow API.

**Option C: Use multi-model approach**
Load separate models for forklift and pallet detection.

---

## 🟡 MEDIUM PROBLEMS (Should Fix)

### Problem #2: Missing Import in Reporter

**Location:** `src/analytics/reporter.py` (line ~170)

**Issue:** The `generate_summary` method imports from `analytics.metrics`:
```python
from analytics.metrics import generate_summary_report
```

This import is inside the method, which is fine, but the import path might fail depending on how the module is loaded.

**Impact:** Potential import error at runtime.

**Solution:** Move import to top of file or use relative import:
```python
from .metrics import generate_summary_report
```

---

### Problem #3: No Error Handling for Video Processing

**Location:** `pipelines/batch_processor.py`

**Issue:** The main processing loop doesn't have comprehensive error handling. If an error occurs mid-video, resources may not be released.

**Impact:** Resource leaks, incomplete processing.

**Solution:** Wrap processing in try-except-finally:

```python
def process_video(self, video_path: str | Path, output_name: Optional[str] = None) -> dict:
    video_path = Path(video_path)
    reader = None
    video_writer = None
    
    try:
        # ... existing processing code ...
    except Exception as e:
        logger.error(f"Error processing {video_path}: {e}")
        raise
    finally:
        if reader:
            reader.release()
        if video_writer:
            video_writer.release()
```

---

### Problem #4: Hysteresis Not Implemented

**Location:** `src/state/classifier.py`

**Issue:** The `rules.yaml` config has hysteresis settings:
```yaml
hysteresis:
  idle_to_moving: 3.0
  moving_to_idle: 0.5
```

But these are not used in the StateClassifier.

**Impact:** State transitions may be too sensitive.

**Solution:** Implement hysteresis in `_get_raw_state`:

```python
def _get_raw_state(
    self,
    velocity: float,
    is_carrying_pallet: bool,
    current_state: ForkliftState = None
) -> ForkliftState:
    """Determine raw state with hysteresis."""
    # Apply hysteresis based on current state
    if current_state in [ForkliftState.MOVING_EMPTY, ForkliftState.MOVING_LOADED]:
        # Use lower threshold to stay moving
        threshold = self.idle_threshold * 0.5  # moving_to_idle factor
    else:
        # Use higher threshold to start moving
        threshold = self.idle_threshold * 1.5  # idle_to_moving factor
    
    is_idle = velocity < threshold
    
    # ... rest of classification logic
```

---

## 🟢 MINOR PROBLEMS (Nice to Fix)

### Problem #5: Hardcoded Cost Values

**Location:** `src/analytics/metrics.py`

**Issue:** Default cost of $75/hour is hardcoded:
```python
def calculate_cost_of_waste(idle_time_seconds: float, cost_per_hour: float = 75.0)
```

**Solution:** Already configurable via parameter, but ensure it's always read from config in `BatchProcessor`.

---

### Problem #6: No GPU Memory Management

**Location:** `src/detection/detector.py`

**Issue:** Long videos may cause GPU OOM as PyTorch caches are not cleared.

**Solution:** Add periodic cache clearing:

```python
def detect_frame(self, frame: np.ndarray, frame_id: int = 0) -> list[Detection]:
    # Clear GPU cache periodically
    if frame_id > 0 and frame_id % 1000 == 0:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ... rest of detection code
```

---

### Problem #7: Missing TrackedObject Initialization in Tests

**Location:** `tests/test_tracker.py`

**Issue:** Test for `TrackedObject` creation is incomplete (truncated at line 100).

**Impact:** Incomplete test coverage.

**Solution:** Complete the test implementation.

---

## 📋 Implementation Checklist

### Critical (Week 1)
- [ ] Fix pallet detection (train model or integrate Roboflow)

### Medium (Week 2)
- [ ] Fix import in reporter.py
- [ ] Add error handling to BatchProcessor
- [ ] Implement hysteresis

### Minor (Week 3+)
- [ ] Ensure config values used everywhere
- [ ] Add GPU memory management
- [ ] Complete test coverage
- [ ] Add type hints throughout

---

## 🔗 File Dependencies

```
process_video.py
    └── batch_processor.py
        ├── detector.py
        │   └── entities.py
        ├── tracker.py (✅ has reset)
        │   └── entities.py
        ├── pallet_detector.py (✅ has analyze_frame)
        │   └── entities.py
        ├── motion_estimator.py (✅ has reset)
        │   └── entities.py
        ├── classifier.py (✅ has reset, ⚠️ needs hysteresis)
        │   └── entities.py
        ├── activity_segmenter.py
        │   └── entities.py
        ├── metrics.py
        │   └── entities.py
        ├── reporter.py (⚠️ import issue)
        │   └── metrics.py
        └── visualizer.py
            └── entities.py
```

---

## 🧪 Testing After Fixes

Run these commands to verify fixes:

```bash
# Unit tests
pytest tests/ -v

# Integration test
python scripts/process_video.py --input data/sample_videos/test.mp4 --output data/outputs --visualize

# Check for import errors
python -c "from pipelines.batch_processor import BatchProcessor; print('OK')"
```

---

*Document created: January 18, 2026*
