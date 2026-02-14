from pydantic import BaseModel


class EmotionResponse(BaseModel):
    probs: list = []
    best_prob: float = -1.0
    predicted_id: int = -1
    predicted_class: str = ""
    predicted_name: str = ""

# Định nghĩa pydatic model
# file xu ly anh
# utils: thu muc tien ich mang tính lặp đi lặp lại
