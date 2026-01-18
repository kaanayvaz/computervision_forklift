# Forklift Idle Time Detection System

Detect forklift idle time and non-value-added activities in warehouse CCTV footage.

## Features

- **YOLO-based Detection**: Detect forklifts, pallets, and people
- **Object Tracking**: Persistent ID tracking with ByteTrack
- **Pallet Detection**: Rule-based "carrying pallet" classification
- **State Classification**: Identify idle vs. active states (rule-based, explainable)
- **Analytics**: Utilization metrics, idle time, cost of waste

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/forklift-analytics.git
cd forklift-analytics

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (optional - will auto-download on first run)
# pip install ultralytics
# yolo detect download model=yolov8s.pt
```

## Quick Start

```bash
# Process a video
python scripts/process_video.py --input data/sample_videos/test.mp4 --output data/outputs --visualize

# Run tests
pytest tests/ -v
```

## Project Structure

```
forklift_analytics/
├── config/               # Configuration files
│   ├── inference.yaml    # Model settings
│   ├── rules.yaml        # Classification rules
│   └── cameras/          # Per-camera calibration
├── src/                  # Source code
│   ├── core/             # Data structures, utilities
│   ├── detection/        # YOLO wrapper
│   ├── tracking/         # ByteTrack integration
│   ├── spatial/          # Pallet-on-forklift logic
│   ├── motion/           # Velocity estimation
│   ├── state/            # Rule-based classification
│   ├── analytics/        # Metrics and reporting
│   └── visualization/    # Output annotation
├── pipelines/            # End-to-end orchestration
├── scripts/              # CLI tools
├── tests/                # Unit and integration tests
├── data/                 # Videos, annotations, outputs
├── models/               # YOLO weights
└── docs/                 # Documentation
```

## Configuration

Edit `config/inference.yaml` for model settings:
```yaml
model:
  weights_path: "models/yolov8s.pt"
  device: "cuda"  # or "cpu"
  
detection:
  confidence_threshold: 0.5
```

Edit `config/rules.yaml` for classification thresholds:
```yaml
motion:
  velocity_idle_threshold: 2.0  # pixels/frame

state:
  idle_duration_threshold: 30  # seconds
```

## Performance Targets

- <100ms per frame processing
- 95% precision on "carrying pallet" detection
- 90% recall on forklift detection

## License

MIT