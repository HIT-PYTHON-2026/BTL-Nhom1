import sys
import torch
import torchvision

from .resnet_model import ResNet18, ResidualBlock
from backend.src.emotion_classification.config.emotion_cfg import EmotionDataConfig
from backend.app.utils import Logger, AppPath, save_cache
from .load_model import download_model
from torch.nn import functional as F
from PIL import Image

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

LOGGER = Logger(__file__, log_file='predictor.log')
LOGGER.log.info('Starting Model Serving')


class Predictor:
    def __init__(self, model_name: str, model_alias: str, device: str = "cpu"):
        self.model_name = model_name
        self.model_alias = model_alias
        self.device = device
        self.load_model()
        self.create_transform()

    async def predict(self, image, image_name):
        pil_img = Image.open(image)
        LOGGER.save_requests(pil_img, image_name)

        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')

        transformed_image = self.transforms_(pil_img).unsqueeze(0)
        output = await self.model_inference(transformed_image)
        probs, best_prob, predicted_id, predicted_class = self.output2pred(
            output)

        LOGGER.log_model(self.model_name, self.model_alias)
        LOGGER.log_response(best_prob, predicted_id, predicted_class)

        torch.cuda.empty_cache()
        save_cache(
            image_name,
            AppPath.CAPTURED_DATA_DIR,
            self.model_name,
            self.model_alias,
            probs,
            best_prob,
            predicted_id,
            predicted_class
        )
        return {
            "probs": probs,
            "best_prob": best_prob,
            "predicted_id": predicted_id,
            "predicted_class": predicted_class,
            "predictor_name": self.model_name,
            "predictor_alias": self.model_alias
        }

    def load_model(self):
        download_model()
        try:
            self.model = ResNet18(
                residual_block=ResidualBlock,
                n_blocks_lst=EmotionDataConfig.N_BLOCK_LST,
                n_classes=EmotionDataConfig.N_CLASSES
            )
            weights_path = AppPath.BACKEND_DIR / 'src' / 'emotion_classification' / \
                'models' / 'weights' / 'emotion_classification_weights.pt'

            checkpoint = torch.load(
                weights_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint.state_dict())

            self.model.to(self.device)
            self.model.eval()
            LOGGER.log.info(f"Model loaded & eval mode: {self.model_name}")
        except Exception as e:
            LOGGER.log.error(f"Load model failed: {e}")
            raise e

    def create_transform(self):
        img_size = EmotionDataConfig.IMG_SIZE
        mean = EmotionDataConfig.NORMALIZE_MEAN
        std = EmotionDataConfig.NORMALIZE_STD

        self.transforms_ = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)
        ])

    async def model_inference(self, input_tensor):
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            output = self.model(input_tensor)
        return output.cpu()

    def output2pred(self, output):
        probabilities = F.softmax(output, dim=1)
        best_prob, predict_id = torch.max(probabilities, 1)
        best_prob = best_prob.item()
        predicted_id = predict_id.item()

        predicted_class = EmotionDataConfig.ID2LABEL[predicted_id]
        probs_list = probabilities.squeeze().tolist()

        return probs_list, best_prob, predicted_id, predicted_class
