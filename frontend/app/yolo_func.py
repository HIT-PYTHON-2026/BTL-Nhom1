import gradio as gr
import requests
from PIL import Image
import numpy as np
import os
from utils import draw_bounding_boxes
from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv("API_URL")


def detect_objects(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    with gr.Blocks():
        with gr.Column():
            temp_filename = "temp_upload.jpg"
            image.save(temp_filename)

            files = {'file_upload': open(temp_filename, 'rb')}

            try:
                url = f"{API_URL}/v1/emotion_classification/detect"
                response = requests.post(url, files=files)
                response.raise_for_status()

                result = response.json()

                pil_image = Image.open(temp_filename).convert("RGB")
                actual_detections = result.get('results', [])

                if actual_detections is None:
                     return pil_image, "No faces detected"
                
                annotated_image = draw_bounding_boxes(
                    pil_image, actual_detections)

                results_text = "Detected Faces:\n"
                for detection in actual_detections:
                    results_text += (
                        f"(Confidence: {detection[-1]:.2f})\n"
                    )

                return annotated_image, results_text

            except requests.RequestException as e:
                return image, f"Error detecting faces: {str(e)}"
            finally:
                if 'file' in files:
                    files['file'].close()
