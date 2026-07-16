
---

### 1.4 src/train.py

```python
"""
YOLOv8 Training Script
Old Building Object Detection
"""

from ultralytics import YOLO

def train_model():
    # Load pretrained model
    model = YOLO('yolov8n.pt')
    
    # Train on custom dataset
    results = model.train(
        data='data/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,  # GPU
        project='runs/detect',
        name='old_building_objects',
        exist_ok=True
    )
    
    print(f"Training complete. Best model saved to: {results.best}")
    return results

if __name__ == '__main__':
    train_model()