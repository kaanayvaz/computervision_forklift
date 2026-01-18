# Environment Configuration Setup

## Overview
This project uses environment variables to securely manage API keys and sensitive configuration.

## Setup Instructions

### 1. Install python-dotenv
```bash
pip install python-dotenv
```

### 2. Create your .env file
Copy the example file and add your credentials:

```bash
cp .env.example .env
```

### 3. Edit .env file
Open `.env` and add your Roboflow API key:

```env
ROBOFLOW_API_KEY=your_actual_api_key_here
ROBOFLOW_WORKSPACE=ss-rz6v8
ROBOFLOW_FORKLIFT_PROJECT=forklift-0jmzj-uvcoy
ROBOFLOW_FORKLIFT_VERSION=1
ROBOFLOW_PALLET_PROJECT=pallet-6awi8-zcqu2
ROBOFLOW_PALLET_VERSION=1
```

### 4. Get Your Roboflow API Key
1. Go to [Roboflow](https://roboflow.com/)
2. Sign in to your account
3. Navigate to Settings → API
4. Copy your API key

## Usage

### In Python Scripts
```python
from core.env_config import get_roboflow_config

# Load configuration
config = get_roboflow_config()
api_key = config["api_key"]
```

### With RoboflowDetector
```python
from detection.roboflow_detector import RoboflowDetector

# Automatically loads from environment
detector = RoboflowDetector.from_env()

# Or pass explicitly
detector = RoboflowDetector(api_key="your_key")
```

### Running Scripts
```bash
# The .env file is automatically loaded
python scripts/process_video_roboflow.py --input video.mp4 --output outputs/
```

## Security Notes

- ✅ `.env` file is added to `.gitignore` and will NOT be committed to git
- ✅ Use `.env.example` as a template (safe to commit)
- ✅ Never hardcode API keys in source code
- ✅ Never commit `.env` file to version control

## Environment Variables Reference

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `ROBOFLOW_API_KEY` | Your Roboflow API key | Yes | - |
| `ROBOFLOW_WORKSPACE` | Roboflow workspace name | No | ss-rz6v8 |
| `ROBOFLOW_FORKLIFT_PROJECT` | Forklift detection project | No | forklift-0jmzj-uvcoy |
| `ROBOFLOW_FORKLIFT_VERSION` | Forklift model version | No | 1 |
| `ROBOFLOW_PALLET_PROJECT` | Pallet detection project | No | pallet-6awi8-zcqu2 |
| `ROBOFLOW_PALLET_VERSION` | Pallet model version | No | 1 |
| `COST_PER_IDLE_HOUR` | Cost calculation for analytics | No | 75.0 |

## Troubleshooting

### "ROBOFLOW_API_KEY not found"
- Make sure `.env` file exists in project root
- Verify `ROBOFLOW_API_KEY=` line is present in `.env`
- Check for typos in variable name

### Import Error: "No module named 'dotenv'"
```bash
pip install python-dotenv
```

### .env file not being read
- Ensure `.env` is in the project root directory
- Check file encoding (should be UTF-8)
- Restart your Python interpreter/IDE

## Alternative: System Environment Variables

You can also set environment variables at the system level:

**Windows PowerShell:**
```powershell
$env:ROBOFLOW_API_KEY = "your_key_here"
```

**Windows CMD:**
```cmd
set ROBOFLOW_API_KEY=your_key_here
```

**Linux/Mac:**
```bash
export ROBOFLOW_API_KEY="your_key_here"
```

## Files Created

- `.env` - Your actual secrets (DO NOT COMMIT)
- `.env.example` - Template file (safe to commit)
- `src/core/env_config.py` - Configuration loader
- `.gitignore` - Updated to exclude `.env`
