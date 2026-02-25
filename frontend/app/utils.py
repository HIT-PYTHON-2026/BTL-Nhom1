import pandas as pd
import numpy as np
import cv2


def draw_bounding_boxes(image, detections):
    image = np.array(image)
    for detection in detections:
        x1, y1, x2, y2, confidence = detection

        color = (0, 255, 0) if float(confidence) > 0.5 else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text = f"Face ({confidence:.2f})"
        cv2.putText(image, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image
