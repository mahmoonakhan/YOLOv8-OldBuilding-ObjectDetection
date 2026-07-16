"""
YOLOv8 Inference Script
Old Building Object Detection
"""

import argparse
from ultralytics import YOLO

def predict(source, weights, save=False, conf=0.25):
    """
    Run inference on image or video.
    
    Args:
        source: Path to image/video
        weights: Path to trained model weights
        save: Save output video
        conf: Confidence threshold
    """
    # Load model
    model = YOLO(weights)
    
    # Run prediction
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        stream=True  # Prevents OOM for large videos
    )
    
    for r in results:
        boxes = r.boxes
        print(f"Detected {len(boxes)} objects")
    
    print(f"Inference complete. Output saved to: runs/detect/predict/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Image or video path')
    parser.add_argument('--weights', type=str, default='models/best.pt', help='Model weights')
    parser.add_argument('--save', action='store_true', help='Save output')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    predict(args.source, args.weights, args.save, args.conf)