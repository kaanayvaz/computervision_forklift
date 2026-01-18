#!/usr/bin/env python
"""
Quick test script to verify Roboflow pipeline components are working.

This script checks:
1. Environment variables are loaded
2. Roboflow API can connect
3. Models can be loaded
4. Pipeline components initialize

Run from project root:
    python scripts/test_roboflow_setup.py
"""

import sys
from pathlib import Path

# Add src and pipelines to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "pipelines"))

def test_env_config():
    """Test environment configuration."""
    print("\n1. Testing Environment Configuration...")
    try:
        from core.env_config import get_roboflow_config
        config = get_roboflow_config()
        
        print(f"   ✅ API Key: {'*' * 10}{config['api_key'][-4:]}")
        print(f"   ✅ Workspace: {config['workspace']}")
        print(f"   ✅ Forklift Project: {config['forklift_project']} v{config['forklift_version']}")
        print(f"   ✅ Pallet Project: {config['pallet_project']} v{config['pallet_version']}")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_roboflow_connection():
    """Test Roboflow API connection."""
    print("\n2. Testing Roboflow API Connection...")
    try:
        from roboflow import Roboflow
        from core.env_config import get_roboflow_config
        
        config = get_roboflow_config()
        rf = Roboflow(api_key=config['api_key'])
        
        print(f"   ✅ Connected to Roboflow")
        return rf, config
    except ImportError:
        print("   ❌ Roboflow not installed. Run: pip install roboflow")
        return None, None
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return None, None

def test_models(rf, config):
    """Test model loading."""
    print("\n3. Testing Model Loading...")
    
    if rf is None:
        print("   ⏭️  Skipping (no connection)")
        return False
    
    try:
        # Test forklift model
        workspace = config.get('workspace')
        if workspace:
            forklift_proj = rf.workspace(workspace).project(config['forklift_project'])
        else:
            forklift_proj = rf.workspace().project(config['forklift_project'])
        forklift_model = forklift_proj.version(config['forklift_version']).model
        print(f"   ✅ Forklift model loaded: {config['forklift_project']}")
        
        # Test pallet model
        if workspace:
            pallet_proj = rf.workspace(workspace).project(config['pallet_project'])
        else:
            pallet_proj = rf.workspace().project(config['pallet_project'])
        pallet_model = pallet_proj.version(config['pallet_version']).model
        print(f"   ✅ Pallet model loaded: {config['pallet_project']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False

def test_pipeline_components():
    """Test pipeline component initialization."""
    print("\n4. Testing Pipeline Components...")
    
    try:
        from tracking.tracker import ForkliftTracker
        tracker = ForkliftTracker()
        print("   ✅ ForkliftTracker initialized")
    except Exception as e:
        print(f"   ❌ ForkliftTracker failed: {e}")
        return False
    
    try:
        from spatial.pallet_detector import SpatialAnalyzer
        analyzer = SpatialAnalyzer()
        print("   ✅ SpatialAnalyzer initialized")
    except Exception as e:
        print(f"   ❌ SpatialAnalyzer failed: {e}")
        return False
    
    try:
        from motion.motion_estimator import MotionEstimator
        estimator = MotionEstimator()
        print("   ✅ MotionEstimator initialized")
    except Exception as e:
        print(f"   ❌ MotionEstimator failed: {e}")
        return False
    
    try:
        from state.classifier import StateClassifier
        classifier = StateClassifier()
        print("   ✅ StateClassifier initialized")
    except Exception as e:
        print(f"   ❌ StateClassifier failed: {e}")
        return False
    
    try:
        from analytics.activity_segmenter import ActivitySegmenter
        segmenter = ActivitySegmenter()
        print("   ✅ ActivitySegmenter initialized")
    except Exception as e:
        print(f"   ❌ ActivitySegmenter failed: {e}")
        return False
    
    try:
        from visualization.visualizer import Visualizer
        visualizer = Visualizer()
        print("   ✅ Visualizer initialized")
    except Exception as e:
        print(f"   ❌ Visualizer failed: {e}")
        return False
    
    return True

def test_roboflow_processor():
    """Test RoboflowBatchProcessor initialization."""
    print("\n5. Testing RoboflowBatchProcessor...")
    
    try:
        from roboflow_batch_processor import RoboflowBatchProcessor
        processor = RoboflowBatchProcessor(visualize=False)
        print("   ✅ RoboflowBatchProcessor initialized")
        return True
    except Exception as e:
        print(f"   ❌ RoboflowBatchProcessor failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ROBOFLOW PIPELINE SETUP TEST")
    print("=" * 60)
    
    results = []
    
    # Test 1: Environment
    results.append(("Environment Config", test_env_config()))
    
    # Test 2: Roboflow Connection
    rf, config = test_roboflow_connection()
    results.append(("Roboflow Connection", rf is not None))
    
    # Test 3: Models
    results.append(("Model Loading", test_models(rf, config)))
    
    # Test 4: Pipeline Components
    results.append(("Pipeline Components", test_pipeline_components()))
    
    # Test 5: Processor
    results.append(("Roboflow Processor", test_roboflow_processor()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! Pipeline is ready to use.")
        print("\nRun a video analysis with:")
        print("   python scripts/process_video_roboflow_integrated.py <video_path>")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
