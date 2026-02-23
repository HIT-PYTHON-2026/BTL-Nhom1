import cv2
import asyncio
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


@router.websocket("/ws")
async def get_stream(websocket: WebSocket):
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
                        cv2.rectangle(frame, (x1 - 50, y1 - 50),
                                      (x2 + 50, y2 + 50), (0, 255, 0), 2)
                        cv2.putText(frame, last_label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        face_img = frame[y1 - 50: y2 + 50, x1 - 50: x2 + 50]

                        if face_img.size > 0:
                            face_img = Image.fromarray(face_img)
                            predictor.create_transform()
                            transform = predictor.transforms_
                            face_tensor = transform(face_img)
                            face_buffer.append(face_tensor)

                            if len(face_buffer) == batch_size:
                                input_batch = torch.stack(
                                    face_buffer).to(device)

                                with torch.no_grad():
                                    outputs = predictor.model(input_batch)
                                    probs = torch.softmax(outputs, dim=1)
                                    avg_probs = torch.mean(probs, dim=0)
                                    max_idx = torch.argmax(avg_probs).item()
                                    last_label = f"{EmotionDataConfig.ID2LABEL[max_idx]} ({avg_probs[max_idx]*100:.1f}%)"

                                face_buffer = []

                ret, buffer = cv2.imencode('.jpg', frame)
                # await websocket.send_text("WEBCAM")
                await websocket.send_bytes(buffer.tobytes())

            await asyncio.sleep(0.03)

    except (WebSocketDisconnect, ConnectionClosed):
        print("Client disconnected")
