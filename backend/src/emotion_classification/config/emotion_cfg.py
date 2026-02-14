import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent))

class EmotionDataConfig():
    N_CLASSES = 7
    IMG_SIZE = 48
    CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise']
    ID2LABEL = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Suprise'}
    LABEL2ID = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Suprise': 6}
    N_BLOCK_LST = [2, 2, 2, 2]
    NORMALIZE_MEAN = {0.485, 0.456, 0.406}
    NORMALIZE_STD = {0.229, 0.224, 0.225}
    
class ModelConfig:
    ROOT_DIR = Path(__file__).parent.parent
    MODEL_NAME = 'ResNet18'
    MODEL_WEIGHT = ROOT_DIR / 'models' / 'weights' / 'emotion_classification_weights.pt'
    DEVICE = 'cpu'