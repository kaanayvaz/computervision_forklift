"""
Process video using Roboflow cloud inference.

This script uses Roboflow's hosted models for forklift and pallet detection.
No local GPU or training required - inference runs in the cloud.

Usage:
    python scripts/process_video_roboflow.py --input VIDEO_PATH --output OUTPUT_DIR

Requirements:
    pip install roboflow
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from core.utils import setup_logging, get_logger
from core.env_config import get_roboflow_config


# Roboflow API configuration from environment variables
try:
    roboflow_config = get_roboflow_config()
    ROBOFLOW_API_KEY = roboflow_config["api_key"]
    FORKLIFT_PROJECT = roboflow_config["forklift_project"]
    PALLET_PROJECT = roboflow_config["pallet_project"]
    WORKSPACE = roboflow_config["workspace"]
except ValueError as e:
    print(f"Error: {e}")
    print("Please set ROBOFLOW_API_KEY in .env file")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process video using Roboflow cloud inference"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/outputs",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second to process (default: 5)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum confidence threshold (default: 0.25)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main processing function."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger = get_logger("roboflow_process")
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input video not found: {input_path}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Roboflow Cloud Video Processing")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {args.output}")
    print(f"FPS:    {args.fps}")
    print(f"Confidence: {args.confidence}")
    print("=" * 60 + "\n")
    
    try:
        from roboflow import Roboflow
        
        # Initialize Roboflow
        logger.info("Connecting to Roboflow...")
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        
        # Load forklift model
        logger.info(f"Loading forklift model: {FORKLIFT_PROJECT}")
        forklift_project = rf.workspace(WORKSPACE).project(FORKLIFT_PROJECT)
        forklift_model = forklift_project.version(1).model
        
        # Load pallet model
        logger.info(f"Loading pallet model: {PALLET_PROJECT}")
        pallet_project = rf.workspace(WORKSPACE).project(PALLET_PROJECT)
        pallet_model = pallet_project.version(1).model
        
        # Process video with forklift model
        print("\n" + "-" * 60)
        print("FORKLIFT DETECTION")
        print("-" * 60)
        logger.info("Starting forklift video inference...")
        
        forklift_job_id, forklift_url, forklift_expire = forklift_model.predict_video(
            str(input_path),
            fps=args.fps,
            prediction_type="batch-video",
        )
        logger.info(f"Job ID: {forklift_job_id}")
        logger.info("Waiting for results (this may take a few minutes)...")
        
        forklift_results = forklift_model.poll_until_video_results(forklift_job_id)
        
        # Count forklift detections
        forklift_count = 0
        if isinstance(forklift_results, dict):
            for frame_key, frame_data in forklift_results.items():
                if isinstance(frame_data, dict) and "predictions" in frame_data:
                    forklift_count += len(frame_data["predictions"])
                elif isinstance(frame_data, list):
                    forklift_count += len(frame_data)
        
        print(f"✓ Forklift detection complete: {forklift_count} detections")
        
        # Process video with pallet model
        print("\n" + "-" * 60)
        print("PALLET DETECTION")
        print("-" * 60)
        
        pallet_count = 0
        pallet_results = {}
        
        try:
            logger.info("Starting pallet video inference...")
            
            pallet_job_id, pallet_url, pallet_expire = pallet_model.predict_video(
                str(input_path),
                fps=args.fps,
                prediction_type="batch-video",
            )
            logger.info(f"Job ID: {pallet_job_id}")
            logger.info("Waiting for results (this may take a few minutes)...")
            
            pallet_results = pallet_model.poll_until_video_results(pallet_job_id)
            
            # Count pallet detections
            if isinstance(pallet_results, dict):
                for frame_key, frame_data in pallet_results.items():
                    if isinstance(frame_data, dict) and "predictions" in frame_data:
                        pallet_count += len(frame_data["predictions"])
                    elif isinstance(frame_data, list):
                        pallet_count += len(frame_data)
            
            print(f"✓ Pallet detection complete: {pallet_count} detections")
            
        except Exception as e:
            logger.warning(f"Pallet detection skipped: {e}")
            print(f"⚠ Pallet detection skipped (model not available)")
        
        # Save results
        output_dir = Path(args.output) / "roboflow_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "video_path": str(input_path),
            "fps": args.fps,
            "confidence": args.confidence,
            "forklift_detections": forklift_results,
            "forklift_count": forklift_count,
            "pallet_detections": pallet_results,
            "pallet_count": pallet_count,
        }
        
        output_file = output_dir / f"{input_path.stem}_detections.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Forklift detections: {forklift_count}")
        print(f"Pallet detections:   {pallet_count}")
        print(f"Results saved to:    {output_file}")
        print("=" * 60)
        
    except ImportError:
        logger.error("Roboflow not installed. Run: pip install roboflow")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
