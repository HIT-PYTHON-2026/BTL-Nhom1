from pathlib import Path


class AppPath:
    BACKEND_DIR = Path(__file__).parent.parent.parent

    LOG_DIR = BACKEND_DIR / "app" / "logs"

    CACHE_DIR = BACKEND_DIR / "cache"
    CAPTURED_DATA_DIR = CACHE_DIR / "capture_data"

    RESNET_MODEL_WEIGHT = BACKEND_DIR / "src" / 'emotion_classification' / \
        'models' / 'weights' / 'emotion_classification_weights.pt'
    YOLO_MODEL_WEIGHT = BACKEND_DIR / "src" / 'emotion_classification' / \
        'models' / 'weights' / 'yolov8n.pt'


AppPath.LOG_DIR.mkdir(parents=True, exist_ok=True)
AppPath.CACHE_DIR.mkdir(parents=True, exist_ok=True)
AppPath.CAPTURED_DATA_DIR.mkdir(parents=True, exist_ok=True)
