import sys
import numpy as np
import cv2
import torch

from typing import List, Tuple, Optional
from PIL import Image
from ultralytics import YOLO
from src.emotion_classification.config.detect_cfg import YoloConfig
from app.utils import Logger, AppPath, save_cache
from torchvision import transforms

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

LOGGER = Logger(__file__, log_file='detector.log')
LOGGER.log.info('Starting Model Serving')


class PersonDetector:
    def __init__(
        self,
        model_name: Optional[Path] = None,
        model_weight: Optional[Path] = None,
        conf_threshold: float = YoloConfig.YOLO_CONFIG_THRESHOLD,
        iou_threshold: float = YoloConfig.YOLO_IOU_THRESHOLD,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.model_weight = model_weight
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._load_model()

    def _load_model(self):
        try:
            if not self.model_weight.exists():
                self.model = YOLO('yolov8n.pt')
                self.model_weight.parent.mkdir(parents=True, exist_ok=True)
            else:
                self.model = YOLO(str(self.model_weight))

            self.model.to(self.device)
            LOGGER.log.info(
                f"Successfully loaded model: {self.model_name} from {self.model_weight}")

        except Exception as e:
            LOGGER.log.error(f"Fail to load model: {str(e)}")
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect_persons(self, image, image_name: str):
        pil_img = Image.open(image)
        LOGGER.save_requests(pil_img, image_name)

        if pil_img.mode == "RGBA":
            pil_img = pil_img.convert("RGB")

        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[YoloConfig.YOLO_PERSON_CLASS_ID],
            verbose=False
        )

        persons = []

        if len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    persons.append((x1, y1, x2, y2, float(conf)))

        LOGGER.log_model(self.model_name)
        LOGGER.log_response(x1, y1, x2, y2, float(conf))

        torch.cuda.empty_cache()

        return persons

    def visualize_detections(
        self,
        image: Image,
        persons: List[Tuple[int, int, int, int, float]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> Image:
        transform = transforms.ToTensor()
        output = transform(image)

        for i, (x1, y1, x2, y2, conf) in enumerate(persons):
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            label = f"Person {i + 1}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(
                output,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            cv2.putText(
                output,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        return output


def test_detector():
    import cv2
    import os
    from datetime import datetime

    print("="*60)
    print("Testing YOLO Person Detector")
    print("="*60)

    detector = PersonDetector(model_weight=AppPath.YOLO_MODEL_WEIGHT)

    save_dir = "captured_images"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible open Camera")
        return

    print("Enter 's to cap, 'q' to escape")
    file_path = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"captured_{timestamp}.jpg"

            file_path = os.path.join(save_dir, file_name)

            cv2.imwrite(file_path, frame)
            print(f"Đã lưu ảnh tại: {file_path}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Running...")
    persons = detector.detect_persons(file_path, "test.jpg")

    print(f"{len(persons)} persons")

    if len(persons) > 0:
        for i, (x1, y1, x2, y2, conf) in enumerate(persons):
            print(
                f"    Person {i + 1}: bbox=({x1}, {y1}, {x2}, {y2}), confidence={conf:.3f}")

    print("\n Yolo detection is working")
    print("="*60)


if __name__ == "__main__":
    test_detector()
