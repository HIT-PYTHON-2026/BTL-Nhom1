import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.emotion_classification.models.emotion_predictor import Predictor
from src.emotion_classification.models.yolo_detector import FacesDetector
from src.emotion_classification.config.emotion_cfg import ModelConfig
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

