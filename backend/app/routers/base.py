# Định nghĩa API Endpoint
from fastapi import APIRouter
from .emotion_router import router as emotion_cls_route
from .stream_router import router as stream_router

router = APIRouter()
router.include_router(emotion_cls_route, prefix="/v1/emotion_classification")
router.include_router(stream_router, prefix="/v1/emotion_classification")

