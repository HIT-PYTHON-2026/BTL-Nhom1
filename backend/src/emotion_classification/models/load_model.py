import gdown
import sys
import os
from ultralytics import YOLO
import requests

from pathlib import Path
sys.path.append(str(Path(__file__).parent))
# https://drive.google.com/file/d/12gb4yspu3eOo-qpCGwAQmXF5SVz6LFBg/view?usp=drive_link
# FILE_ID = "1ojjbBCgGtWeBIJHjWj5J7DEMdngkItDD"
FILE_ID = "12gb4yspu3eOo-qpCGwAQmXF5SVz6LFBg"
CURRENT_DIR = Path(__file__).parent
WEIGHTS_DIR = CURRENT_DIR / 'weights'
RESNET_MODEL_WEIGHT = WEIGHTS_DIR/'emotion_classification_weights.pt'
YOLO_MODEL_WEIGHT = WEIGHTS_DIR/'yolov8n-face-lindevs.pt'


def resnet_download():
    RESNET_MODEL_WEIGHT.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(RESNET_MODEL_WEIGHT):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        # 2. Thuc hien tai ve
        gdown.download(url, str(RESNET_MODEL_WEIGHT), quiet=False, fuzzy=True)
        print(f"Downloaded: {RESNET_MODEL_WEIGHT}")
        return True
    else:
        print(f"Model is already exsits.")
        return False


def yoloface_download():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt"
    if not YOLO_MODEL_WEIGHT.exists():
        print(f"Đang tải model về: {YOLO_MODEL_WEIGHT}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(YOLO_MODEL_WEIGHT, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Tải model thành công!")
        else:
            print(f"Lỗi tải file: Mã lỗi {response.status_code}")
    else:
        print("Model đã tồn tại, bỏ qua bước tải.")


if __name__ == "__main__":
   resnet_download()
   yoloface_download()