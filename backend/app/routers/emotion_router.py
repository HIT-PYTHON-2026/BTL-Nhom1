from src.emotion_classification.models.emotion_predictor import Predictor
from src.emotion_classification.config.emotion_cfg import ModelConfig
from schemas.emotion_schema import EmotionResponse
from fastapi import APIRouter
from fastapi import File, UploadFile
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


router = APIRouter()
predictor = Predictor(
    model_name=ModelConfig.MODEL_NAME,
    model_weight=ModelConfig.MODEL_WEIGHT,
    device=ModelConfig.DEVICE
)


@router.post('/predict')
async def predict(file_upload: UploadFile = File(...)):
    response = await predictor.predict(
        image=file_upload.file,
        image_name=file_upload.filename
    )
    return EmotionResponse(**response)
