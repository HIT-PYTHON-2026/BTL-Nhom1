import gradio as gr
import cv2
import time
import os
from yolo_func import detect_objects
from resnet_func import classify_emotion


with gr.Blocks() as demo:
    gr.Markdown("# Demo Emotion Classification")

    with gr.Tab("Detections"):
        gr.Markdown("Face detection using YOLOV8")
        with gr.Row():
            input_image = gr.Image(
                type="pil", label="Upload Image", height=450, width=900)
        with gr.Row():
            detect_btn = gr.Button("Detect Faces")
        with gr.Row():
            annotated_image = gr.Image(label="Annotated Image")
        with gr.Row():
            detection_results = gr.Textbox(label="Detection Results")

        detect_btn.click(
            fn=detect_objects,
            inputs=input_image,
            outputs=[annotated_image, detection_results]
        )
    with gr.Tab("Classification"):
        gr.Markdown("Emotion classification using ResNet18")
        with gr.Row():
            input_image = gr.Image(
                type="pil", label="Upload Image", height=450, width=900)
        with gr.Row():
            detect_btn = gr.Button("Classify Emotion")
        with gr.Row():
            classification_results = gr.Textbox(label="Classification Results")

        detect_btn.click(
            fn=classify_emotion,
            inputs=input_image,
            outputs=[classification_results]
        )

demo.launch(server_name="127.0.0.1", server_port=8000)
