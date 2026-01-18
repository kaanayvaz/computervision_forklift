# Forklift Analytics - Documentation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Video Input                                   │
│                    (CCTV Footage .mp4/.avi)                         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Video Reader                                   │
│              (Frame extraction, resize, skip)                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Forklift Detector (YOLO)                          │
│        Detect: Forklifts, Pallets, People                           │
│        Output: List[Detection] per frame                            │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Object Tracker (ByteTrack)                        │
│        Assign persistent IDs to detections                          │
│        Output: List[TrackedObject]                                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Spatial Analyzer                                  │
│        Determine: Is forklift carrying pallet?                      │
│        Rules: IoU, containment, position                            │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Motion Estimator                                  │
│        Calculate: Velocity from bbox displacement                   │
│        Apply: Temporal smoothing                                    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              State Classifier (Rule-Based)                           │
│        States: IDLE, MOVING_EMPTY, MOVING_LOADED, LOADING           │
│        NO ML - Explainable rules only                               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Activity Segmenter                                │
│        Group frames into activity segments                          │
│        Identify non-value-added activities                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Analytics & Reporting                             │
│        Calculate: Utilization, idle time, cost of waste            │
│        Output: JSON, CSV, annotated video                           │
└─────────────────────────────────────────────────────────────────────┘
```

## State Classification Rules

| State | Velocity | Carrying Pallet | Description |
|-------|----------|-----------------|-------------|
| IDLE | < threshold | No | Stationary, not productive |
| MOVING_EMPTY | ≥ threshold | No | Moving to pick up load |
| MOVING_LOADED | ≥ threshold | Yes | Transporting load |
| LOADING | < threshold | Transitioning No→Yes | Picking up pallet |
| UNLOADING | < threshold | Transitioning Yes→No | Dropping pallet |

## Non-Value-Added Activities

1. **Idle Waiting** - Forklift stationary > 30s without load operation
2. **Blocked** - Forklift stationary with another forklift nearby
3. **Operator Absent** - No person detected near stationary forklift > 60s

## Configuration Files

- `config/inference.yaml` - Model and detection settings
- `config/rules.yaml` - State classification rules
- `config/cameras/*.yaml` - Per-camera calibration
