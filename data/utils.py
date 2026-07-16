"""
Utility functions for YOLOv8 project
"""

import cv2
import numpy as np
from ultralytics import YOLO

def draw_custom_boxes(image, results, class_colors=None):
    """
    Draw bounding boxes with custom styling.
    """
    if class_colors is None:
        class_colors = {
            0: (0, 255, 0),    # Door - Green
            1: (255, 0, 0),    # Dustbin - Blue
            2: (0, 0, 255),    # Outdoor Plant - Red
            3: (255, 255, 0)   # Window - Cyan
        }
    
    class_names = ['Door', 'Dustbin', 'Outdoor Plant', 'Window']
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)
        conf = float(box.conf)
        
        color = class_colors.get(cls, (128, 128, 128))
        label = f"{class_names[cls]} {conf:.2f}"
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def get_model_info(model_path):
    """
    Get model metadata.
    """
    model = YOLO(model_path)
    info = {
        'task': model.task,
        'names': model.names,
        'nc': len(model.names)
    }
    return info