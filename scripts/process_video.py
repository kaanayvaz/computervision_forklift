"""
Main entry point for processing forklift videos.

Usage:
    python scripts/process_video.py --input VIDEO_PATH --output OUTPUT_DIR
    
Options:
    --input, -i     Path to input video file
    --output, -o    Output directory for results
    --config, -c    Path to config directory (default: config/)
    --visualize     Generate annotated output video
    --verbose, -v   Enable verbose logging
"""

import argparse
import sys
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from core.utils import setup_logging, load_config, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process forklift video for idle time detection"
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
        "--config", "-c",
        type=str,
        default="config",
        help="Path to config directory"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate annotated output video"
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
    logger = get_logger("process_video")
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input video not found: {input_path}")
        sys.exit(1)
    
    # Validate config directory
    config_dir = Path(args.config)
    if not (config_dir / "inference.yaml").exists():
        logger.error(f"Config not found: {config_dir / 'inference.yaml'}")
        sys.exit(1)
    
    logger.info(f"Processing video: {input_path}")
    logger.info(f"Output directory: {args.output}")
    
    print("\n" + "=" * 60)
    print("Forklift Analytics - Video Processor")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {args.output}")
    print(f"Config: {config_dir}")
    print("=" * 60 + "\n")
    
    try:
        # Import and run the batch processor
        from pipelines.batch_processor import BatchProcessor
        
        # Initialize processor
        processor = BatchProcessor(
            config_dir=args.config,
            output_dir=args.output,
            visualize=args.visualize
        )
        
        # Process the video
        results = processor.process_video(input_path)
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Forklifts tracked: {results['track_count']}")
        print(f"Average FPS: {results['avg_fps']:.1f}")
        print(f"Utilization: {results['analytics'].utilization_percentage:.1f}%")
        print(f"Idle time: {results['analytics'].total_idle_time_seconds / 60:.1f} minutes")
        print(f"Cost of waste: ${results['analytics'].cost_of_waste:.2f}")
        print("=" * 60)
        print(f"\nReports saved to: {args.output}/reports/")
        if args.visualize:
            print(f"Annotated video saved to: {args.output}/videos/")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
