import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent))

class YoloConfig:
    YOLO_CONFIG_THRESHOLD = 0.5
    YOLO_IOU_THRESHOLD = 0.45
    YOLO_PERSON_CLASS_ID = 0
    YOLO_IMAGE_SIZE = 640
    
    FACE_SCALE_FACTOR = 1.1
    FACE_MIN_NEIGHBORS = 5
    FACE_MIN_SIZE = (30, 30)
    FACE_PADDING = 10
    
    MAX_IMAGE_SIZE = 1920
    