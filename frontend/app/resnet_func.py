import gradio as gr
import requests
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv("API_URL")


def classify_emotion(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    with gr.Blocks():
        with gr.Column():
            temp_filename = "temp_upload.jpg"
            image.save(temp_filename)

            files = {'file_upload': open(temp_filename, 'rb')}

            try:
                url = f"{API_URL}/v1/emotion_classification/predict"
                response = requests.post(url, files=files)
                response.raise_for_status()

                result = response.json()

                results_text = "Classified Faces:\n"
                results_text += str(result['predicted_class'])
                
                return results_text

            except requests.RequestException as e:
                return image, f"Error classifying emotion: {str(e)}"
            finally:
                if 'file' in files:
                    files['file'].close()
