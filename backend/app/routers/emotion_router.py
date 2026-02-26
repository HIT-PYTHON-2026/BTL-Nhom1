import sys
import io
import numpy as np
import cv2
import torch
from torch.nn import functional as F
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from PIL import Image
from src.emotion_classification.models.emotion_predictor import Predictor
from src.emotion_classification.models.yolo_detector import FacesDetector
from src.emotion_classification.config.emotion_cfg import ModelConfig, EmotionDataConfig
from app.schemas.emotion_schema import EmotionResponse
from app.schemas.face_schema import FaceResponse
from fastapi import APIRouter
from fastapi import File, UploadFile


router = APIRouter()
predictor = Predictor(
    model_name=ModelConfig.MODEL_NAME,
    model_weight=ModelConfig.MODEL_WEIGHT,
    device=ModelConfig.DEVICE
)

detector = FacesDetector(
    model_name="yolov8n-face-lindevs",
)


@router.post('/predict')
async def predict(file_upload: UploadFile = File(...)):
    response = await predictor.predict(
        image=file_upload.file,
        image_name=file_upload.filename
    )
    return EmotionResponse(**response)

@router.post('/detect')
async def detectFace(file_upload: UploadFile = File(...)):
    response = await detector.detect_faces(
        image=file_upload.file,
        image_name=file_upload.filename
    )
    data_to_response = {
        "status": "success",
        "message": "Detection completed",
        "face_count": len(response),
        "results": response
    }
    return FaceResponse(**data_to_response)


@router.post('/analyze')
async def analyze_faces(file_upload: UploadFile = File(...)):
    """
    Detect all faces → crop each → predict emotion per face.
    Returns combined bounding boxes + per-face emotion results.
    """
    image_bytes = await file_upload.read()
    pil_img = Image.open(io.BytesIO(image_bytes))

    if pil_img.mode == 'RGBA':
        pil_img = pil_img.convert('RGB')

    # Convert to numpy for OpenCV operations
    img_np = np.array(pil_img)

    # Step 1: Detect faces
    results = detector.model.predict(
        pil_img,
        conf=detector.conf_threshold,
        iou=detector.iou_threshold,
        verbose=False
    )

    faces_data = []

    if len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            h, w = img_np.shape[:2]

            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = map(int, box)

                # Padded crop (30px)
                pad = 30
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(w, x2 + pad)
                cy2 = min(h, y2 + pad)

                face_crop = img_np[cy1:cy2, cx1:cx2]

                if face_crop.size == 0:
                    continue

                # Step 2: Predict emotion for this face
                face_pil = Image.fromarray(face_crop)
                face_tensor = predictor.transforms_(face_pil).unsqueeze(0)

                with torch.no_grad():
                    output = predictor.model(face_tensor.to(predictor.device))
                    probs = F.softmax(output, dim=1)
                    probs_list = probs.squeeze().tolist()
                    best_prob, pred_id = torch.max(probs, 1)
                    predicted_class = EmotionDataConfig.ID2LABEL[pred_id.item()]

                faces_data.append({
                    "face_id": i + 1,
                    "box": [x1, y1, x2, y2],
                    "confidence": round(float(conf), 3),
                    "predicted_class": predicted_class,
                    "best_prob": round(best_prob.item(), 4),
                    "probs": [round(p, 4) for p in probs_list],
                })

    return {
        "face_count": len(faces_data),
        "faces": faces_data,
        "predictor_name": predictor.model_name,
    }
