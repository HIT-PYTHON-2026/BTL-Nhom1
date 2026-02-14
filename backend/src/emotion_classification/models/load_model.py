import gdown
import sys
import os

from pathlib import Path
sys.path.append(str(Path(__file__).parent))

FILE_ID = "1f_9ANnfO21YV-jv4PL9KB7TR1XqU9E02" 
BACKEND_DIR = Path(__file__).parent.parent
MODEL_WEIGHT = BACKEND_DIR/'models'/'weights'/'emotion_classification_weights.pt'

def download_model():
    MODEL_WEIGHT.parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(MODEL_WEIGHT):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        # 2. Thuc hien tai ve
        gdown.download(url, str(MODEL_WEIGHT), quiet=False, fuzzy=True)
        print(f"Downloaded: {MODEL_WEIGHT}")
        return True
    else:
        print(f"Model is already exsits.")
        return False


if __name__ == "__main__":
    download_model()
