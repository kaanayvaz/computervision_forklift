"""
Pallet tracking using Roboflow cloud inference.

This script specifically tracks pallets in warehouse video footage using
Roboflow's batch video processing API.

Usage:
    python scripts/track_pallets_roboflow.py --input VIDEO_PATH --output OUTPUT_DIR

Requirements:
    pip install roboflow
"""

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from core.utils import setup_logging, get_logger
from core.env_config import get_roboflow_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Track pallets in video using Roboflow cloud inference"
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
        default="data/outputs/pallet_tracking",
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
    
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save raw JSON results"
    )
    
    return parser.parse_args()


def process_pallet_video(
    video_path: str,
    fps: int,
    confidence: float,
    logger
) -> dict:
    """
    Process video for pallet detection using Roboflow.
    
    Args:
        video_path: Path to input video
        fps: Frames per second to process
        confidence: Minimum confidence threshold
        logger: Logger instance
        
    Returns:
        Dictionary with detection results
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("roboflow package not installed. Install with: pip install roboflow")
        sys.exit(1)
    
    # Get configuration
    config = get_roboflow_config()
    
    logger.info("Initializing Roboflow API...")
    rf = Roboflow(api_key=config["api_key"])
    
    # Load pallet detection project
    logger.info(f"Loading pallet project: {config['pallet_project']}")
    project = rf.workspace(config["workspace"]).project(config["pallet_project"])
    model = project.version(config["pallet_version"]).model
    
    logger.info(f"Starting batch video processing...")
    logger.info(f"Video: {video_path}")
    logger.info(f"FPS: {fps}, Confidence: {confidence}")
    
    # Start video prediction
    job_id, signed_url, expire_time = model.predict_video(
        video_path,
        fps=fps,
        prediction_type="batch-video",
    )
    
    logger.info(f"Job submitted: {job_id}")
    logger.info(f"Signed URL: {signed_url}")
    logger.info(f"Expires: {expire_time}")
    logger.info("Waiting for results (this may take a few minutes)...")
    
    # Poll for results
    results = model.poll_until_video_results(job_id)
    
    logger.info("✅ Processing complete!")
    
    return {
        "job_id": job_id,
        "signed_url": signed_url,
        "expire_time": expire_time,
        "results": results,
        "config": {
            "fps": fps,
            "confidence": confidence,
            "project": config["pallet_project"],
            "version": config["pallet_version"],
        }
    }


def save_results(results: dict, output_dir: Path, video_name: str, save_json: bool, logger):
    """Save processing results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary_path = output_dir / f"{video_name}_{timestamp}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PALLET TRACKING RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Video: {video_name}\n")
        f.write(f"Job ID: {results['job_id']}\n")
        f.write(f"FPS: {results['config']['fps']}\n")
        f.write(f"Confidence: {results['config']['confidence']}\n")
        f.write(f"Project: {results['config']['project']}\n")
        f.write(f"Version: {results['config']['version']}\n\n")
        
        # Count detections
        total_frames = 0
        total_detections = 0
        
        if isinstance(results['results'], dict):
            for frame_id, frame_data in results['results'].items():
                total_frames += 1
                if 'predictions' in frame_data:
                    total_detections += len(frame_data['predictions'])
        
        f.write(f"Total Frames Processed: {total_frames}\n")
        f.write(f"Total Pallet Detections: {total_detections}\n")
        f.write(f"Average Detections/Frame: {total_detections/total_frames:.2f}\n" if total_frames > 0 else "")
        f.write("\n" + "=" * 60 + "\n")
    
    logger.info(f"Summary saved: {summary_path}")
    
    # Save full JSON if requested
    if save_json:
        json_path = output_dir / f"{video_name}_{timestamp}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Full results saved: {json_path}")
    
    # Create CSV of detections
    csv_path = output_dir / f"{video_name}_{timestamp}_detections.csv"
    with open(csv_path, 'w') as f:
        f.write("frame_id,detection_id,class,confidence,x,y,width,height,x_center,y_center\n")
        
        if isinstance(results['results'], dict):
            for frame_id, frame_data in results['results'].items():
                if 'predictions' in frame_data:
                    for idx, pred in enumerate(frame_data['predictions']):
                        f.write(f"{frame_id},{idx},{pred.get('class', 'pallet')},")
                        f.write(f"{pred.get('confidence', 0)},")
                        f.write(f"{pred.get('x', 0)},{pred.get('y', 0)},")
                        f.write(f"{pred.get('width', 0)},{pred.get('height', 0)},")
                        f.write(f"{pred.get('x', 0)},{pred.get('y', 0)}\n")
    
    logger.info(f"Detections CSV saved: {csv_path}")
    
    return summary_path, csv_path


def print_summary(results: dict):
    """Print results summary to console."""
    print("\n" + "=" * 60)
    print("PALLET TRACKING RESULTS")
    print("=" * 60)
    print(f"Job ID: {results['job_id']}")
    print(f"FPS: {results['config']['fps']}")
    print(f"Confidence: {results['config']['confidence']}")
    
    # Count detections
    total_frames = 0
    total_detections = 0
    
    if isinstance(results['results'], dict):
        for frame_id, frame_data in results['results'].items():
            total_frames += 1
            if 'predictions' in frame_data:
                total_detections += len(frame_data['predictions'])
    
    print(f"\nFrames Processed: {total_frames}")
    print(f"Total Pallet Detections: {total_detections}")
    if total_frames > 0:
        print(f"Average Detections/Frame: {total_detections/total_frames:.2f}")
    
    print("\nSigned Video URL:")
    print(results['signed_url'])
    print(f"\nExpires: {results['expire_time']}")
    print("=" * 60 + "\n")


def main():
    """Main processing function."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger = get_logger("pallet_tracker")
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input video not found: {input_path}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Pallet Tracking - Roboflow Cloud Processing")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {args.output}")
    print(f"FPS:    {args.fps}")
    print("=" * 60 + "\n")
    
    try:
        # Process video
        results = process_pallet_video(
            str(input_path),
            args.fps,
            args.confidence,
            logger
        )
        
        # Print summary
        print_summary(results)
        
        # Save results
        output_dir = Path(args.output)
        video_name = input_path.stem
        
        summary_path, csv_path = save_results(
            results,
            output_dir,
            video_name,
            args.save_json,
            logger
        )
        
        print("\n✅ Processing Complete!")
        print(f"Results saved to: {output_dir}")
        print(f"  - Summary: {summary_path.name}")
        print(f"  - CSV: {csv_path.name}")
        if args.save_json:
            print(f"  - JSON: {video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_results.json")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Install required packages: pip install roboflow")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
