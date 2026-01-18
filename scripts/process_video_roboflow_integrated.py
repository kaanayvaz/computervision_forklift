#!/usr/bin/env python
"""
Process video using Roboflow cloud detection for forklifts and pallets.

This script provides the full analytics pipeline:
1. Roboflow cloud detection (forklift + pallet models)
2. Object tracking with ByteTrack
3. Pallet carrying detection via spatial analysis
4. State classification (IDLE, MOVING_EMPTY, MOVING_LOADED, etc.)
5. Activity segmentation
6. Analytics generation (utilization, idle time, cost of waste)
7. Report generation (JSON, CSV, summary)

Usage:
    python scripts/process_video_roboflow_integrated.py <video_path>
    python scripts/process_video_roboflow_integrated.py <video_path> --fps 5 --confidence 0.3

Environment:
    Requires ROBOFLOW_API_KEY in .env file

Output:
    - Annotated video: data/outputs/videos/<name>_annotated.mp4
    - JSON report: data/outputs/reports/<name>.json
    - CSV export: data/outputs/reports/<name>.csv
    - Summary: data/outputs/reports/<name>_summary.txt
"""

import sys
import argparse
from pathlib import Path

# Add src and pipelines to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "pipelines"))

from roboflow_batch_processor import RoboflowBatchProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process forklift video with Roboflow cloud detection and full analytics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Process with default settings (5 FPS, 0.25 confidence):
        python scripts/process_video_roboflow_integrated.py data/sample_videos/warehouse.mp4

    Process at higher frame rate:
        python scripts/process_video_roboflow_integrated.py video.mp4 --fps 10

    Process with stricter confidence threshold:
        python scripts/process_video_roboflow_integrated.py video.mp4 --confidence 0.4

    Skip visualization (faster processing):
        python scripts/process_video_roboflow_integrated.py video.mp4 --no-visualize

Environment Variables (in .env file):
    ROBOFLOW_API_KEY       - Your Roboflow API key
    ROBOFLOW_FORKLIFT_PROJECT - Forklift detection project ID
    ROBOFLOW_PALLET_PROJECT   - Pallet detection project ID
        """
    )
    
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for Roboflow processing (default: 5)"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )
    
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip generating annotated output video"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs",
        help="Output directory for results (default: data/outputs)"
    )
    
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Configuration directory (default: config)"
    )
    
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom name for output files (default: video filename)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("FORKLIFT NON-VALUE-ADDED TIME ANALYSIS")
    print("Using Roboflow Cloud Detection")
    print("=" * 70)
    print(f"\nInput Video: {video_path}")
    print(f"Processing FPS: {args.fps}")
    print(f"Confidence Threshold: {args.confidence}")
    print(f"Generate Video: {not args.no_visualize}")
    print(f"Output Directory: {args.output_dir}")
    print()
    
    # Create processor
    processor = RoboflowBatchProcessor(
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        visualize=not args.no_visualize,
        fps=args.fps,
        confidence=args.confidence
    )
    
    # Process video
    try:
        results = processor.process_video(
            video_path,
            output_name=args.output_name
        )
        
        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE")
        print("=" * 70)
        
        # Summary
        analytics = results["analytics"]
        print(f"\n📊 Summary:")
        print(f"   Forklifts Tracked: {results['track_count']}")
        print(f"   Activities Detected: {len(results['activities'])}")
        print(f"   Utilization Rate: {analytics.utilization_percentage:.1f}%")
        print(f"   Total Idle Time: {analytics.total_idle_time_seconds / 60:.1f} minutes")
        print(f"   Estimated Cost of Waste: ${analytics.cost_of_waste:.2f}")
        
        print(f"\n📁 Output Files:")
        print(f"   Reports: {args.output_dir}/reports/")
        if not args.no_visualize:
            print(f"   Video: {args.output_dir}/videos/")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n❌ Processing interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
