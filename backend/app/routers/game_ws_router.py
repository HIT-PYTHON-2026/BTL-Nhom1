"""
WebSocket endpoint cho Game Emotion Express.
Nhận base64 frame từ frontend → YOLO detect mặt → ResNet18 predict cảm xúc → trả JSON.
"""
import base64
import json
import asyncio
import numpy as np
import cv2
import torch

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed
from PIL import Image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.emotion_classification.models.emotion_predictor import Predictor
from src.emotion_classification.models.yolo_detector import FacesDetector
from src.emotion_classification.config.emotion_cfg import EmotionDataConfig
from utils.app_path import AppPath

router = APIRouter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping 7 backend labels → 3 game emotions
LABEL_TO_GAME_EMOTION = {
    "Happy": "happy",
    "Sad": "sad",
    "Suprise": "surprised",
    # Các emotion sau không map sang game emotion nào
    "Angry": None,
    "Disgust": None,
    "Fear": None,
    "Neutral": None,
}


@router.websocket("/game-ws")
async def game_emotion_ws(websocket: WebSocket):
    """
    WebSocket endpoint cho game.
    - Nhận: base64 encoded JPEG frame từ frontend
    - Trả: JSON { face_detected, emotion, confidence, raw_label }
    """
    await websocket.accept()

    # Khởi tạo models (1 lần khi kết nối)
    detector = FacesDetector(model_weight=AppPath.YOLO_MODEL_WEIGHT)
    predictor = Predictor(
        model_name="ResNet18",
        model_weight=AppPath.RESNET_MODEL_WEIGHT,
        device="cpu"
    )

    face_buffer = []
    batch_size = 3  # Nhỏ hơn stream_router (5) để phản hồi nhanh hơn cho game
    last_result = {
        "face_detected": False,
        "emotion": None,
        "confidence": 0.0,
        "raw_label": None,
    }

    try:
        while True:
            # Nhận base64 frame từ frontend
            data = await websocket.receive_text()

            # Decode base64 → numpy array → OpenCV frame
            try:
                # Frontend gửi: "data:image/jpeg;base64,/9j/4AAQ..." hoặc raw base64
                if "," in data:
                    data = data.split(",", 1)[1]

                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_text(json.dumps({
                        "face_detected": False,
                        "emotion": None,
                        "confidence": 0.0,
                        "raw_label": None,
                        "error": "Invalid frame"
                    }))
                    continue

            except Exception as e:
                await websocket.send_text(json.dumps({
                    "face_detected": False,
                    "emotion": None,
                    "confidence": 0.0,
                    "raw_label": None,
                    "error": str(e)
                }))
                continue

            # YOLO face detection
            results = detector.model.predict(
                frame,
                conf=detector.conf_threshold,
                iou=detector.iou_threshold,
                verbose=False
            )

            face_found = False

            if len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()

                    # Lấy khuôn mặt lớn nhất (confidence cao nhất)
                    best_idx = np.argmax(confidences)
                    box = boxes[best_idx]
                    x1, y1, x2, y2 = map(int, box)

                    # Mở rộng vùng crop (padding 30px)
                    h, w = frame.shape[:2]
                    pad = 30
                    x1_p = max(0, x1 - pad)
                    y1_p = max(0, y1 - pad)
                    x2_p = min(w, x2 + pad)
                    y2_p = min(h, y2 + pad)

                    face_img = frame[y1_p:y2_p, x1_p:x2_p]

                    if face_img.size > 0:
                        face_found = True
                        # Convert sang PIL → transform → tensor
                        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                        face_tensor = predictor.transforms_(face_pil)
                        face_buffer.append(face_tensor)

                        # Khi đủ batch → predict
                        if len(face_buffer) >= batch_size:
                            input_batch = torch.stack(face_buffer).to(device)

                            with torch.no_grad():
                                outputs = predictor.model(input_batch)
                                probs = torch.softmax(outputs, dim=1)
                                avg_probs = torch.mean(probs, dim=0)
                                max_idx = torch.argmax(avg_probs).item()
                                confidence = avg_probs[max_idx].item()
                                raw_label = EmotionDataConfig.ID2LABEL[max_idx]
                                game_emotion = LABEL_TO_GAME_EMOTION.get(raw_label)

                            last_result = {
                                "face_detected": True,
                                "emotion": game_emotion,
                                "confidence": round(confidence, 3),
                                "raw_label": raw_label,
                            }
                            face_buffer = []

            if not face_found:
                # Không tìm thấy mặt → reset buffer, trả trạng thái cuối
                face_buffer = []
                last_result = {
                    "face_detected": False,
                    "emotion": None,
                    "confidence": 0.0,
                    "raw_label": None,
                }

            # Luôn gửi kết quả mới nhất về frontend
            await websocket.send_text(json.dumps(last_result))

    except (WebSocketDisconnect, ConnectionClosed):
        print("[Game WS] Client disconnected")
    except Exception as e:
        print(f"[Game WS] Error: {e}")
