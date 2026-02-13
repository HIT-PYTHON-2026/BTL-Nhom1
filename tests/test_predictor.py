import asyncio
import torch
import sys
from backend.src.emotion_classification.models import emotion_predictor # Thay 'predictor' bằng tên file của bạn

from pathlib import Path
sys.path.append(str(Path(__file__).parent))

async def main():
    # 1. Khởi tạo Predictor
    # Kiểm tra xem GPU có sẵn không để truyền vào
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Đang khởi tạo Predictor trên: {device} ---")
    
    try:
        predictor = Predictor(
            model_name="ResNet18_Emotion", 
            model_alias="v1", 
            device=device
        )
        print("✅ Khởi tạo và load model thành công!")
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return

    # 2. Tạo một ảnh giả (hoặc trỏ tới file ảnh thật) để test
    # Nếu bạn có file ảnh thật, hãy thay bằng đường dẫn: image_path = "test.jpg"
    print("\n--- Đang chạy thử dự đoán ---")
    image_path = "test_image.jpg"
    
    # Tạo ảnh ảo nếu không có file thật để tránh lỗi FileNotFoundError khi test code
    from PIL import Image
    import io
    img = Image.new('RGB', (224, 224), color = 'red')
    img.save(image_path)

    try:
        # Giả lập request ảnh (đọc file dưới dạng binary giống như nhận từ API)
        with open(image_path, "rb") as f:
            image_bytes = io.BytesIO(f.read())
            result = await predictor.predict(image_bytes, "test_image.jpg")
            
        print("\n--- Kết quả dự đoán ---")
        print(f"Dự đoán là: {result['predicted_class']} (ID: {result['predicted_id']})")
        print(f"Độ tin cậy (Prob): {result['best_prob']:.4f}")
        print(f"Danh sách xác suất: {result['probs']}")
        
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán: {e}")

if __name__ == "__main__":
    asyncio.run(main())