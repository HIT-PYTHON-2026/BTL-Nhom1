import sys
import torch
import torchvision

from .resnet_model import ResNet18, ResidualBlock
from src.emotion_classification.config.emotion_cfg import EmotionDataConfig
from app.utils import Logger, AppPath, save_cache
from .load_model import download_model
from torch.nn import functional as F
from PIL import Image

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

LOGGER = Logger(__file__, log_file='predictor.log')
LOGGER.log.info('Starting Model Serving')


class Predictor:
    def __init__(self, model_name: str, model_weight: str, device: str = "cpu"):
        self.model_name = model_name
        self.model_weight = model_weight
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

        LOGGER.log_model(self.model_name)
        LOGGER.log_response(best_prob, predicted_id, predicted_class)

        torch.cuda.empty_cache()
        save_cache(
            image_name,
            AppPath.CAPTURED_DATA_DIR,
            self.model_name,
            self.model_weight,
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
            "predictor_weight": self.model_weight
        }

    def load_model(self):
        try:
            self.model = ResNet18(
                residual_block=ResidualBlock,
                n_blocks_lst=EmotionDataConfig.N_BLOCK_LST,
                n_classes=EmotionDataConfig.N_CLASSES
            )
            
            checkpoint = torch.load(
                AppPath.MODEL_WEIGHT, # Đảm bảo AppPath đúng
                map_location=self.device,
                weights_only=False 
            )
            
            if isinstance(checkpoint, torch.nn.Module):
                state_dict = checkpoint.state_dict()
            else:
                state_dict = checkpoint
              
            self.model.load_state_dict(state_dict, strict=False)

            self.model.to(self.device)
            self.model.eval()

            LOGGER.log.info(
                f"Successfully loaded model: {self.model_name} from {self.model_weight}")
        except Exception as e:
            LOGGER.log.error(f"Fail to load model: {str(e)}")
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
