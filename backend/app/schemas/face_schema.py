from pydantic import BaseModel
from typing import Optional, List, Tuple

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    
class FaceResult(BaseModel):
    face_id: int
    confidence: float
    box: BoundingBox

class FaceResponse(BaseModel):
    results: List[Tuple[int, int, int, int, float]]
    # status: str = "success"
    # message: Optional[str] = None
    # face_count: int
    # results: List[FaceResult]
    