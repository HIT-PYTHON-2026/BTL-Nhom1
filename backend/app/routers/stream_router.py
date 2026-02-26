import cv2
import asyncio
import base64
import numpy as np
import torch

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from websockets.exceptions import ConnectionClosed
from PIL import Image
from ultralytics import YOLO
from utils.app_path import AppPath
from src.emotion_classification.models.emotion_predictor import Predictor
from src.emotion_classification.models.yolo_detector import FacesDetector
from src.emotion_classification.config.emotion_cfg import EmotionDataConfig

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

router = APIRouter()
camera = cv2.VideoCapture(0)
templates = Jinja2Templates(directory="templates")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@router.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _process_frame(frame, detector, predictor, face_buffer, last_label, batch_size):
    """Shared face detection + emotion prediction logic for both server and client camera."""
    results = detector.model.predict(
        frame,
        conf=detector.conf_threshold,
        iou=detector.iou_threshold,
        verbose=False
    )

    if len(results) > 0:
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            h, w = frame.shape[:2]
            # Padded bounding box
            x1_p = max(0, x1 - 50)
            y1_p = max(0, y1 - 50)
            x2_p = min(w, x2 + 50)
            y2_p = min(h, y2 + 50)

            cv2.rectangle(frame, (x1_p, y1_p), (x2_p, y2_p), (0, 255, 0), 2)
            cv2.putText(frame, last_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            face_img = frame[y1_p:y2_p, x1_p:x2_p]

            if face_img.size > 0:
                face_img = Image.fromarray(face_img)
                predictor.create_transform()
                transform = predictor.transforms_
                face_tensor = transform(face_img)
                face_buffer.append(face_tensor)

                if len(face_buffer) == batch_size:
                    input_batch = torch.stack(face_buffer).to(device)

                    with torch.no_grad():
                        outputs = predictor.model(input_batch)
                        probs = torch.softmax(outputs, dim=1)
                        avg_probs = torch.mean(probs, dim=0)
                        max_idx = torch.argmax(avg_probs).item()
                        last_label = f"{EmotionDataConfig.ID2LABEL[max_idx]} ({avg_probs[max_idx]*100:.1f}%)"

                    face_buffer.clear()

    return frame, face_buffer, last_label


@router.websocket("/ws")
async def get_stream(websocket: WebSocket):
    """Server camera: reads from cv2.VideoCapture on the server machine."""
    await websocket.accept()

    detector = FacesDetector(model_weight=AppPath.YOLO_MODEL_WEIGHT)
    predictor = Predictor(
        model_name="ResNet18",
        model_weight=AppPath.RESNET_MODEL_WEIGHT,
        device="cpu"
    )

    face_buffer = []
    last_label = "Initializing..."
    batch_size = 5

    try:
        while True:
            success, frame = camera.read()

            if not success:
                break
            else:
                frame, face_buffer, last_label = _process_frame(
                    frame, detector, predictor, face_buffer, last_label, batch_size
                )

                ret, buffer = cv2.imencode('.jpg', frame)
                await websocket.send_bytes(buffer.tobytes())

            await asyncio.sleep(0.03)

    except (WebSocketDisconnect, ConnectionClosed):
        print("Client disconnected")


@router.websocket("/ws-client")
async def get_stream_client(websocket: WebSocket):
    """Client camera: receives base64 JPEG frames from the browser webcam,
    processes them, and sends back annotated JPEG bytes."""
    await websocket.accept()

    detector = FacesDetector(model_weight=AppPath.YOLO_MODEL_WEIGHT)
    predictor = Predictor(
        model_name="ResNet18",
        model_weight=AppPath.RESNET_MODEL_WEIGHT,
        device="cpu"
    )

    face_buffer = []
    last_label = "Initializing..."
    batch_size = 5

    try:
        while True:
            data = await websocket.receive_text()

            # Strip data URI prefix if present
            if "," in data:
                data = data.split(",", 1)[1]

            try:
                img_bytes = base64.b64decode(data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue
            except Exception:
                continue

            frame, face_buffer, last_label = _process_frame(
                frame, detector, predictor, face_buffer, last_label, batch_size
            )

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                await websocket.send_bytes(buffer.tobytes())

    except (WebSocketDisconnect, ConnectionClosed):
        print("[Client Camera] Client disconnected")
    except Exception as e:
        print(f"[Client Camera] Error: {e}")
