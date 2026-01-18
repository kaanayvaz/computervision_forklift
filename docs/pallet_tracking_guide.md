# 📦 Pallet Tracking with Roboflow

## Overview
This implementation tracks pallets in warehouse video footage using Roboflow's cloud-based batch video processing API.

---

## 🚀 Quick Start

### Basic Usage
```bash
python scripts/track_pallets_roboflow.py --input data/sample_videos/source.mp4 --output data/outputs/pallet_tracking --fps 5
```

### With Full Options
```bash
python scripts/track_pallets_roboflow.py \
    --input data/sample_videos/source.mp4 \
    --output data/outputs/pallet_tracking \
    --fps 5 \
    --confidence 0.25 \
    --save-json \
    --verbose
```

---

## 📋 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Path to input video file | Required |
| `--output`, `-o` | Output directory for results | `data/outputs/pallet_tracking` |
| `--fps` | Frames per second to process | `5` |
| `--confidence` | Minimum detection confidence (0-1) | `0.25` |
| `--save-json` | Save full JSON results | `False` |
| `--verbose`, `-v` | Enable debug logging | `False` |

---

## 📤 Output Files

The script generates three types of output files:

### 1. Summary File (`*_summary.txt`)
```
============================================================
PALLET TRACKING RESULTS
============================================================

Video: source
Job ID: abc123def456...
FPS: 5
Confidence: 0.25
Project: pallet-unicd-k2rg0
Version: 1

Total Frames Processed: 150
Total Pallet Detections: 342
Average Detections/Frame: 2.28

============================================================
```

### 2. Detections CSV (`*_detections.csv`)
```csv
frame_id,detection_id,class,confidence,x,y,width,height,x_center,y_center
1,0,pallet,0.87,150,200,80,60,190,230
1,1,pallet,0.92,450,180,75,58,487,209
2,0,pallet,0.89,152,201,81,61,192,231
...
```

### 3. Full JSON Results (`*_results.json`) - Optional
Complete detection data with all metadata from Roboflow API.

---

## 🔧 How It Works

```python
# 1. Load API configuration from environment
config = get_roboflow_config()
rf = Roboflow(api_key=config["api_key"])

# 2. Load pallet detection model
project = rf.workspace(config["workspace"]).project(config["pallet_project"])
model = project.version(config["pallet_version"]).model

# 3. Submit video for batch processing
job_id, signed_url, expire_time = model.predict_video(
    video_path,
    fps=5,
    prediction_type="batch-video",
)

# 4. Poll for results
results = model.poll_until_video_results(job_id)

# 5. Process and save results
# - Summary text file
# - CSV of detections
# - Optional full JSON
```

---

## 📊 Detection Data Structure

Each detection contains:
- **frame_id**: Frame number in the video
- **class**: Object class (usually "pallet" or "wood pallet")
- **confidence**: Detection confidence (0-1)
- **x, y**: Top-left corner coordinates
- **width, height**: Bounding box dimensions
- **x_center, y_center**: Center point coordinates

---

## 🔐 Configuration

The script uses environment variables from `.env`:

```env
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_WORKSPACE=your_workspace  # Optional - uses default if not set
ROBOFLOW_PALLET_PROJECT=pallet-unicd-k2rg0
ROBOFLOW_PALLET_VERSION=1
```

---

## 📈 Performance Tips

### FPS Selection
- **FPS 1-2**: Slow-moving objects, longer videos (faster processing)
- **FPS 5**: Balanced - good for general warehouse footage
- **FPS 10-15**: Fast-moving objects, need more detail

### Confidence Threshold
- **0.15-0.25**: More detections, may include false positives
- **0.25-0.50**: Balanced (recommended)
- **0.50-0.80**: High confidence only, may miss some pallets

---

## 💡 Example Workflows

### 1. Basic Pallet Counting
```bash
python scripts/track_pallets_roboflow.py -i video.mp4 -o results/ --fps 5
# Check *_summary.txt for total count
```

### 2. Detailed Analysis with JSON
```bash
python scripts/track_pallets_roboflow.py -i video.mp4 -o results/ --save-json
# Analyze full JSON for custom processing
```

### 3. High-Speed Footage
```bash
python scripts/track_pallets_roboflow.py -i video.mp4 --fps 15 --confidence 0.3
# Higher FPS for fast-moving pallets
```

---

## 🔗 Integration with Full Pipeline

To integrate pallet detections with forklift tracking:

1. Run pallet tracking:
```bash
python scripts/track_pallets_roboflow.py -i video.mp4 -o pallet_results/
```

2. Use CSV output in spatial analyzer:
```python
from spatial.pallet_detector import SpatialAnalyzer
import pandas as pd

# Load pallet detections
pallets_df = pd.read_csv("pallet_results/video_detections.csv")

# Process with forklift data
analyzer = SpatialAnalyzer()
# ... combine with forklift tracking
```

---

## 🐛 Troubleshooting

### "ROBOFLOW_API_KEY not found"
- Ensure `.env` file exists in project root
- Check `ROBOFLOW_API_KEY=` is set in `.env`

### "roboflow package not installed"
```bash
pip install roboflow
```

### Job Takes Too Long
- Cloud processing time depends on video length and FPS
- Typical: 2-5 minutes for 1 minute of video at 5 FPS
- Check Roboflow dashboard for job status

### No Pallets Detected
- Try lowering `--confidence` threshold (e.g., `0.15`)
- Verify video contains visible pallets
- Check FPS isn't too low (missing frames)

---

## 📚 Additional Resources

- **Roboflow Documentation**: https://docs.roboflow.com/
- **Batch Video API**: https://docs.roboflow.com/inference/hosted-api/batch-video-api
- **Environment Setup**: See [ENV_SETUP.md](../ENV_SETUP.md)

---

**Last Updated:** January 18, 2026
