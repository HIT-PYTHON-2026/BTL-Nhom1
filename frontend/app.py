import os
import requests
import gradio as gr
import io
import argparse
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv("API_URL")

NUM_CLASSES = 7
ID2CLASS = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
            3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Suprise'}


def predict_api(image_path):
    try:
        image = Image.open(image_path)
        img_name = image_path.split("/")[-1]
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        url = f"{API_URL}/v1/emotion_classification/predict"
        files = {'file_upload': (img_name, img_byte_arr, 'image/jpeg')}
        headers = {'accept': 'application/json'}

        response = requests.post(url, headers=headers, files=files)

        if response.status_code == 200:
            json_results = response.json()
            confidences = {ID2CLASS[i]: json_results["probs"][i]
                           for i in range(NUM_CLASSES)}
            return confidences, json_results
        else:
            error_msg = f"API Error: {response.status_code}"
            return {"Error": 1.0}, {"detail": error_msg}
    except Exception as e:
        print(f"LOG: {str(e)}")
        return {"System error": 1.0}, {"Error detail": str(e)}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--share", action="store_true")
    args = argparser.parse_args()

    interface = gr.Interface(
        fn=predict_api,
        inputs=gr.Image(
            type='filepath', label='Upload Image',
            height=450, width=900
        ),
        outputs=[
            gr.Label(num_top_classes=7, label="Probabilities"),
            gr.JSON(label="Info Output")
        ],
        title="Image Prediction API",
        examples=[
            os.path.join(os.path.dirname(__file__),
                         "images/examples/happy.jpg"),
            os.path.join(os.path.dirname(__file__),
                         "images/examples/suprise.jpg")
        ],
        description="Upload an image to get predictions from the API.")
ui_host = os.getenv("Frontend_HOST")
ui_port = int(os.getenv("Frontend_PORT"))
interface.launch(server_name=ui_host, server_port=ui_port, share=args.share)
