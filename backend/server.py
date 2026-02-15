import uvicorn
from src.emotion_classification.models.resnet_model import ResNet18, ResidualBlock

Resnet = ResNet18
Block = ResidualBlock

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
