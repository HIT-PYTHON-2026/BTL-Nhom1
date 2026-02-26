import uvicorn
from src.emotion_classification.models.resnet_model import ResNet, Block
from src.emotion_classification.models.load_model import resnet_download, yoloface_download

Resnet = ResNet

if __name__ == "__main__":
    resnet_download()
    yoloface_download()
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
