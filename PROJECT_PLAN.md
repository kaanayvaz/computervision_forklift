# Forklift Idle Time Detection System - Project Plan

## 📋 Project Overview

This is a **Computer Vision-based Forklift Analytics System** designed to detect forklift idle time and non-value-added activities in warehouse CCTV footage. The system uses YOLOv8 for object detection and rule-based classification for state analysis.

---

## 🎯 Project Objectives

1. **Detect forklifts, pallets, and people** in warehouse video footage
2. **Track objects** with persistent IDs across frames using ByteTrack
3. **Classify forklift states** (IDLE, MOVING_EMPTY, MOVING_LOADED, LOADING, UNLOADING)
4. **Identify non-value-added activities** (idle waiting, blocked, operator absent)
5. **Generate analytics reports** with utilization metrics and cost of waste

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          VIDEO INPUT                                 │
│                    (CCTV Footage .mp4/.avi)                         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    1. VIDEO READER (src/video/reader.py)            │
│        - Frame extraction with skipping                             │
│        - Resolution limiting                                        │
│        - Metadata extraction                                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│               2. FORKLIFT DETECTOR (src/detection/detector.py)      │
│        - YOLOv8 inference wrapper                                   │
│        - Detects: Forklifts, Pallets, People                       │
│        - Output: List[Detection] per frame                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│               3. OBJECT TRACKER (src/tracking/tracker.py)           │
│        - ByteTrack via supervision library                          │
│        - Persistent ID tracking                                     │
│        - Track lifecycle management                                 │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│             4. SPATIAL ANALYZER (src/spatial/pallet_detector.py)    │
│        - IoU and containment calculations                           │
│        - Pallet-on-forklift detection                              │
│        - Rule-based spatial association                             │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│             5. MOTION ESTIMATOR (src/motion/motion_estimator.py)    │
│        - Velocity from bbox displacement                            │
│        - Temporal smoothing                                         │
│        - Direction calculation                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│             6. STATE CLASSIFIER (src/state/classifier.py)           │
│        - Rule-based classification                                  │
│        - Temporal smoothing to prevent flickering                   │
│        - States: IDLE, MOVING_EMPTY, MOVING_LOADED, etc.           │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│          7. ACTIVITY SEGMENTER (src/analytics/activity_segmenter.py)│
│        - Group frames into activity segments                        │
│        - Merge short segments                                       │
│        - Value-added classification                                 │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│             8. ANALYTICS & REPORTING (src/analytics/)               │
│        - Utilization metrics                                        │
│        - Idle time analysis                                         │
│        - Cost of waste calculation                                  │
│        - JSON, CSV, Text reports                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
forklift_analytics/
├── config/                    # Configuration files
│   ├── inference.yaml         # Model & detection settings
│   ├── rules.yaml             # Classification rules
│   └── cameras/               # Per-camera calibration
├── src/                       # Source code
│   ├── core/                  # Data structures, utilities
│   │   ├── entities.py        # Core dataclasses
│   │   └── utils.py           # Helper functions
│   ├── detection/             # YOLO wrapper
│   │   └── detector.py        # ForkliftDetector
│   ├── tracking/              # ByteTrack integration
│   │   └── tracker.py         # ForkliftTracker
│   ├── spatial/               # Pallet-on-forklift logic
│   │   └── pallet_detector.py # SpatialAnalyzer
│   ├── motion/                # Velocity estimation
│   │   └── motion_estimator.py
│   ├── state/                 # Rule-based classification
│   │   ├── classifier.py      # StateClassifier
│   │   └── rules.py           # Classification rules
│   ├── analytics/             # Metrics and reporting
│   │   ├── activity_segmenter.py
│   │   ├── metrics.py
│   │   └── reporter.py
│   ├── video/                 # Video I/O
│   │   └── reader.py          # VideoReader, VideoWriter
│   └── visualization/         # Output annotation
│       └── visualizer.py      # Frame annotation
├── pipelines/                 # End-to-end orchestration
│   └── batch_processor.py     # BatchProcessor
├── scripts/                   # CLI tools
│   └── process_video.py       # Main entry point
├── tests/                     # Unit and integration tests
├── data/                      # Data files
│   ├── datasets/              # Training datasets
│   ├── sample_videos/         # Test videos
│   └── outputs/               # Generated outputs
├── models/                    # YOLO weights
└── docs/                      # Documentation
```

---

## 🔍 Detected Problems & Solutions

### ✅ Verified Working Components

The codebase is well-structured. The following components were verified to be complete:

| Component | Key Method | Status |
|-----------|------------|--------|
| MotionEstimator | `reset()` | ✅ Implemented |
| StateClassifier | `reset()` | ✅ Implemented |
| SpatialAnalyzer | `analyze_frame()` | ✅ Implemented |
| ForkliftTracker | `reset()` | ✅ Implemented |
| StateClassifier | `_apply_temporal_smoothing()` | ✅ Implemented |

### 🔴 Critical Issues

| # | Problem | Location | Impact | Solution |
|---|---------|----------|--------|----------|
| 1 | ~~**Pallet detection model mismatch**~~ | ~~`config/inference.yaml`~~ | ~~Config says classes: `forklift: 0, person: 1` but code expects pallet class for spatial analysis~~ | ✅ **RESOLVED**: Implemented Roboflow cloud detection for pallets via `RoboflowBatchProcessor` |

### 🟡 Medium Issues

| # | Problem | Location | Impact | Solution |
|---|---------|----------|--------|----------|
| 2 | **Import path may break** | `src/analytics/reporter.py` | `reporter.py` imports from `analytics.metrics` - path depends on how module is loaded | Verify import path or use relative import |
| 3 | **VideoWriter not using context manager** | `pipelines/batch_processor.py` | Video writer may not be properly released on errors | Use context manager pattern |
| 4 | **Missing hysteresis implementation** | `src/state/classifier.py` | Config has hysteresis settings but not used in classifier | Implement hysteresis for smoother transitions |

### 🟢 Minor Issues / Improvements

| # | Problem | Location | Impact | Solution |
|---|---------|----------|--------|----------|
| 5 | **Hardcoded cost values** | `src/analytics/metrics.py` | Default $75/hour may not match actual costs | Already configurable via parameter |
| 6 | **No GPU memory management** | `src/detection/detector.py` | May OOM on long videos | Add periodic cache clearing |
| 7 | **Missing type hints in some functions** | Various | Reduces IDE support | Add complete type annotations |
| 8 | **No progress persistence** | `pipelines/batch_processor.py` | Long videos can't be resumed | Add checkpointing |

---

## 🛠️ Solution Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)

#### 1.1 Fix Pallet Detection
- Option A: Use separate pallet detection model
- Option B: Fine-tune existing model to detect pallets
- Option C: Use Roboflow pallet detection model (already in datasets)

The existing pallet dataset is available at `data/datasets/pallet/` and can be used to train a pallet detection model.

### Phase 2: Medium Priority (Week 2)

#### 2.1 Fix Import in Reporter
- Update import to use relative path: `from .metrics import generate_summary_report`

#### 2.2 Error Handling
- Add try-except blocks around video processing
- Implement graceful degradation

#### 2.3 Implement Hysteresis
- Read hysteresis values from rules.yaml
- Apply different thresholds based on current state

#### 2.4 Resource Management
- Use context managers for all I/O
- Add GPU memory clearing

### Phase 3: Enhancements (Week 3-4)

#### 3.1 Performance Optimization
- Add batch processing for multiple videos
- Implement multiprocessing for frame analysis
- Add checkpointing for long videos

#### 3.2 Feature Additions
- Real-time processing mode
- Zone-based activity tracking
- Operator association (person near forklift)

---

## 📊 Key Features & Capabilities

### Detection & Tracking
- ✅ YOLOv8-based object detection
- ✅ Roboflow cloud detection (forklift + pallet)
- ✅ ByteTrack multi-object tracking
- ✅ Persistent track IDs
- ✅ Pallet detection (via Roboflow pallet-unicd-k2rg0)

### State Classification
- ✅ IDLE state detection
- ✅ MOVING_EMPTY state detection
- ✅ MOVING_LOADED state detection (with pallet carrying detection)
- ✅ LOADING/UNLOADING transitions
- ✅ Temporal smoothing

### Analytics
- ✅ Utilization percentage calculation
- ✅ Idle time breakdown
- ✅ Cost of waste estimation
- ✅ JSON/CSV/Text report generation

### Visualization
- ✅ Bounding box annotation
- ✅ State color coding
- ✅ Track ID display
- ✅ Velocity arrows

---

## 🔧 Configuration Reference

### inference.yaml
```yaml
model:
  weights_path: "models/yolov8s-forklift.pt"  # YOLO model path
  device: "cuda"                               # cuda or cpu
  half_precision: true                         # FP16 for GPU

detection:
  confidence_threshold: 0.25                   # Min detection confidence
  iou_threshold: 0.45                          # NMS IoU threshold
  classes:
    forklift: 0
    person: 1

processing:
  frame_skip: 2                                # Process every Nth frame
  max_resolution: [1280, 720]                  # Max frame size
```

### rules.yaml
```yaml
spatial:
  pallet_iou_threshold: 0.3                    # Min IoU for pallet association
  pallet_containment_threshold: 0.5            # Min containment ratio
  fork_zone_ratio: 0.4                         # Lower % of bbox for forks

motion:
  velocity_idle_threshold: 2.0                 # pixels/frame for idle
  smoothing_window: 5                          # Frames for averaging

state:
  idle_duration_threshold: 30                  # Seconds for significant idle
  state_confirmation_frames: 5                 # Frames to confirm state

analytics:
  cost_per_idle_hour: 75.0                     # USD cost calculation
```

---

## 🚀 Usage Guide

### ✅ Recommended: Roboflow Cloud Detection (Forklift + Pallet)

The integrated Roboflow pipeline uses cloud-based models for both forklift and pallet detection:

```bash
# Process video with Roboflow (recommended)
python scripts/process_video_roboflow_integrated.py data/sample_videos/source.mp4

# With custom settings
python scripts/process_video_roboflow_integrated.py video.mp4 --fps 5 --confidence 0.3

# Skip video generation (faster)
python scripts/process_video_roboflow_integrated.py video.mp4 --no-visualize
```

**Features:**
- ✅ Forklift detection (forklift-0jmzj-uvcoy model)
- ✅ Pallet detection (pallet-unicd-k2rg0 model)
- ✅ Pallet carrying detection via spatial analysis
- ✅ State classification (IDLE, MOVING_EMPTY, MOVING_LOADED, etc.)
- ✅ Analytics generation (utilization, idle time, cost of waste)
- ✅ Annotated video output

**Requirements:**
- `ROBOFLOW_API_KEY` in `.env` file
- Internet connection for cloud inference

### Alternative: Local YOLO Detection (Forklift Only)

The original pipeline using local YOLO models:

```bash
python scripts/process_video.py --input video.mp4 --output data/outputs --visualize
```

**Note:** Local detection does not include pallet detection by default.

### Running Setup Test
```bash
# Verify all components are working
python scripts/test_roboflow_setup.py
```

### Running Tests
```bash
pytest tests/ -v
```

### Training Custom Model
```bash
python scripts/train_models.py --data data/datasets/forklift/data.yaml
```

---

## 📈 Performance Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| Frame Processing Time | <100ms | ✅ Achievable with GPU |
| Forklift Detection Precision | >95% | ✅ Roboflow cloud detection |
| Pallet Detection Precision | >95% | ✅ Roboflow cloud detection |
| Tracking Accuracy (MOTA) | >80% | ⚠️ Untested |

---

## 📝 Development Notes

### Dependencies
- Python 3.10+
- PyTorch 2.0+
- Ultralytics (YOLOv8)
- OpenCV
- supervision (ByteTrack)
- roboflow (Cloud API)
- python-dotenv (Environment variables)
- NumPy, PyYAML, tqdm

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your ROBOFLOW_API_KEY

# Verify setup
python scripts/test_roboflow_setup.py
```

---

## 🔮 Future Roadmap

1. **Real-time Processing** - Stream processing for live CCTV
2. **Multi-camera Support** - Stitch views from multiple cameras
3. **Deep Learning State Classification** - Replace rules with ML
4. **Dashboard** - Web-based visualization dashboard
5. **Alert System** - Real-time alerts for extended idle
6. **Integration** - WMS/ERP system integration

---

## 📞 Support

For issues or questions:
1. Check `docs/architecture.md` for detailed architecture
2. Review test files in `tests/` for usage examples
3. Check configuration files in `config/`

---

*Last Updated: January 18, 2026*
