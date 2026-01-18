"""
Training script for forklift and pallet detection models.

This script trains two YOLOv8 models:
1. Forklift detection model
2. Pallet detection model

Run with: python scripts/train_models.py

Training on CPU takes 2-4 hours per model.
Training on GPU (CUDA) takes 30-60 minutes per model.
"""

from pathlib import Path


def train_forklift_model():
    """Train forklift detection model."""
    from ultralytics import YOLO
    
    print("=" * 60)
    print("TRAINING FORKLIFT DETECTION MODEL")
    print("=" * 60)
    
    # Load base model
    model = YOLO('yolov8s.pt')
    
    # Train
    results = model.train(
        data='data/datasets/forklift/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        name='forklift_custom',
        project='runs/train',
        patience=10,
        workers=2,
        verbose=True
    )
    
    # Save best model to models folder
    best_model_path = Path('runs/train/forklift_custom/weights/best.pt')
    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, 'models/forklift_trained.pt')
        print(f"\n✓ Forklift model saved to: models/forklift_trained.pt")
    
    return results


def train_pallet_model():
    """Train pallet detection model."""
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("TRAINING PALLET DETECTION MODEL")
    print("=" * 60)
    
    # Load base model
    model = YOLO('yolov8s.pt')
    
    # Train
    results = model.train(
        data='data/datasets/pallet/data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        name='pallet_custom',
        project='runs/train',
        patience=10,
        workers=2,
        verbose=True
    )
    
    # Save best model to models folder
    best_model_path = Path('runs/train/pallet_custom/weights/best.pt')
    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, 'models/pallet_trained.pt')
        print(f"\n✓ Pallet model saved to: models/pallet_trained.pt")
    
    return results


if __name__ == "__main__":
    import torch
    
    print("\n" + "=" * 60)
    print("YOLO MODEL TRAINING")
    print("=" * 60)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("\n⚠️  WARNING: Training on CPU is SLOW!")
        print("   Expected time: 2-4 hours per model")
        print("   Consider using Google Colab with GPU for faster training.")
    else:
        print("\n✓ GPU available - training will be faster")
        print("   Expected time: 30-60 minutes per model")
    
    print("\n" + "-" * 60)
    
    # Train both models
    try:
        train_forklift_model()
        train_pallet_model()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print("\nTrained models saved to:")
        print("  - models/forklift_trained.pt")
        print("  - models/pallet_trained.pt")
        print("\nUpdate config/inference.yaml to use the new models:")
        print('  weights_path: "models/forklift_trained.pt"')
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise
