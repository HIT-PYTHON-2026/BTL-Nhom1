import asyncio
import torch
import sys
import os
from pathlib import Path

# Đảm bảo nhận diện được folder 'backend' và 'config'
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Import chính xác Class từ file
from backend.src.emotion_classification.models.emotion_predictor import Predictor

async def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Đang khởi tạo Predictor trên: {device} ---")
    
    try:
        # Khởi tạo instance
        predictor_inst = Predictor(
            model_name="ResNet18_Emotion", 
            model_alias="v1", 
            device=device
        )
        print("✅ Khởi tạo và load model thành công!")
        
        # Chạy thử dự đoán
        print("\n--- Đang chạy thử dự đoán ---")
        image_path = "test_image.jpg"
        
        from PIL import Image
        import io
        img = Image.new('RGB', (224, 224), color = 'red')
        img.save(image_path)

        with open(image_path, "rb") as f:
            image_bytes = io.BytesIO(f.read())
            result = await predictor_inst.predict(image_bytes, "test_image.jpg")
            
        print("\n--- Kết quả dự đoán ---")
        print(f"Dự đoán là: {result['predicted_class']} (ID: {result['predicted_id']})")
        print(f"Độ tin cậy (Prob): {result['best_prob']:.4f}")
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình thực thi: {e}")

if __name__ == "__main__":
    asyncio.run(main())