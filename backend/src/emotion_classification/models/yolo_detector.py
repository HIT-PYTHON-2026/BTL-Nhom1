from pathlib import Path
import io
import sys
import numpy as np
import cv2
import torch

from typing import List, Tuple, Optional
from PIL import Image
from ultralytics import YOLO
from src.emotion_classification.config.detect_cfg import YoloConfig
from src.emotion_classification.config.emotion_cfg import EmotionDataConfig
from app.utils import Logger, AppPath, save_cache
from torchvision import transforms
from .emotion_predictor import Predictor
from .resnet_model import ResNet, Block
Resnet = ResNet

sys.path.append(str(Path(__file__).parent.parent.parent))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGGER = Logger(__file__, log_file='detector.log')
LOGGER.log.info('Starting Model Serving')


class FacesDetector:
    def __init__(
        self,
        model_name: Optional[Path] = None,
        model_weight: Optional[Path] = "src/emotion_classification/models/weights/yolov8n-face-lindevs.pt",
        conf_threshold: float = YoloConfig.YOLO_CONFIG_THRESHOLD,
        iou_threshold: float = YoloConfig.YOLO_IOU_THRESHOLD,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.model_weight = Path(model_weight)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._load_model()

    def _load_model(self):
        try:
            if not self.model_weight.exists():
                self.model = YOLO('yolov8n-face-lindevs.pt')
                self.model_weight.parent.mkdir(parents=True, exist_ok=True)
            else:
                self.model = YOLO(str(self.model_weight))

            self.model.to(self.device)
            # LOGGER.log.info(
            #     f"Successfully loaded model: {self.model_name} from {self.model_weight}")

        except Exception as e:
            # LOGGER.log.error(f"Fail to load model: {str(e)}")
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    async def detect_faces(
        self,
        image,
        image_name: str
    ):
        image = image.read()
        pil_img = Image.open(io.BytesIO(image))


        if pil_img.mode == "RGBA":
            pil_img = pil_img.convert("RGB")

        results = self.model.predict(
            pil_img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[YoloConfig.YOLO_PERSON_CLASS_ID],
            verbose=False
        )

        faces = []

        if len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    faces.append((x1, y1, x2, y2, float(conf)))

        # LOGGER.log_model(self.model_name)
        # LOGGER.log_response(x1, y1, x2, y2, float(conf))

        # torch.cuda.empty_cache()

        return faces

    # def visualize_detections(
    #     self,
    #     image: Image,
    #     faces: List[Tuple[int, int, int, int, float]],
    #     color: Tuple[int, int, int] = (0, 255, 0),
    #     thickness: int = 2
    # ) -> Image:
    #     transform = transforms.ToTensor()
    #     output = transform(image)

    #     for i, (x1, y1, x2, y2, conf) in enumerate(faces):
    #         cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

    #         label = f"Person {i + 1}: {conf:.2f}"
    #         label_size, _ = cv2.getTextSize(
    #             label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    #         cv2.rectangle(
    #             output,
    #             (x1, y1 - label_size[1] - 10),
    #             (x1 + label_size[0], y1),
        #         color,
        #         -1
        #     )

        #     cv2.putText(
        #         output,
        #         label,
        #         (x1, y1 - 5),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (255, 255, 255),
        #         1
        #     )
        # return output


async def test_detector_realtime():
    print("="*60)
    print("Testing YOLO Face Detector REALTIME")
    print("="*60)

    detector = FacesDetector(model_weight=AppPath.YOLO_MODEL_WEIGHT)
    predictor = Predictor(
        model_name="ResNet18",
        model_weight=AppPath.RESNET_MODEL_WEIGHT,
        device="cpu"
    )

    face_buffer = []
    last_label = "Initializing..."
    batch_size = 5

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở Camera")
        return

    print("Nhấn 'q' để thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.model.predict(
            frame,
            conf=detector.conf_threshold,
            iou=detector.iou_threshold,
            classes=[YoloConfig.YOLO_PERSON_CLASS_ID],
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
                        input_batch = torch.stack(face_buffer).to(device)

                        with torch.no_grad():
                            outputs = predictor.model(input_batch)
                            probs = torch.softmax(outputs, dim=1)
                            avg_probs = torch.mean(probs, dim=0)
                            max_idx = torch.argmax(avg_probs).item()
                            last_label = f"{EmotionDataConfig.ID2LABEL[max_idx]} ({avg_probs[max_idx]*100:.1f}%)"

                        face_buffer = []

        cv2.imshow('Realtime Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_detector_realtime())
