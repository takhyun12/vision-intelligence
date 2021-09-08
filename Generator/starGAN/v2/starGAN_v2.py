import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from core.model import Generator, StyleEncoder
import warnings
warnings.filterwarnings('ignore')


class StarGAN_v2:
    def __init__(self, **kwargs: dict) -> None:
        """ Architecture of style transfer network """
        super(StarGAN_v2, self).__init__()
        self.model_name: str = 'StarGAN_v2'
        self.model_version: str = '1.0.0'
        self.source_image_path: str = kwargs['source_image_path']
        self.reference_image_path: str = kwargs['reference_image_path']
        self.pre_trained_model_path: str = kwargs['pre_trained_model_path']

        # configuration
        self.image_size: int = 512
        style_dimensions: int = 768
        domain_number: int = 3

        # model build
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.style_encoder = StyleEncoder(self.image_size, style_dimensions, domain_number).to(self.device)
        self.generator = Generator(self.image_size, style_dimensions, w_hpf=0).to(self.device)

        pre_trained_model = torch.load(self.pre_trained_model_path, map_location=self.device)
        self.generator.load_state_dict(pre_trained_model["generator"])
        self.generator.eval()  # evaluate model

        self.style_encoder.load_state_dict(pre_trained_model["style_encoder"])
        self.style_encoder.eval()  # evaluate model
        del pre_trained_model  # memory cleansing

        # build transform
        self.to_tensor = self.build_transform()

        # load image
        self.source_image_tensor = self.load_image_to_tensor(self.source_image_path)
        self.reference_image_tensor = self.load_image_to_tensor(reference_image_path)

    def generate_new_image(self):
        # show original images
        self.show_tensor_to_pil(self.source_image_tensor[0])
        self.show_tensor_to_pil(self.reference_image_tensor[0])

        # generate
        with torch.no_grad():
            source_style = self.style_encoder(self.source_image_tensor.to(self.device), torch.LongTensor([1]).to(self.device))
            reference_style = self.style_encoder(self.reference_image_tensor.to(self.device), torch.LongTensor([0]).to(self.device))

            generated_source = self.generator(self.source_image_tensor.to(self.device), reference_style)
            generated_reference = self.generator(self.reference_image_tensor.to(self.device), source_style)

            # show generated images
            self.show_tensor_to_pil(generated_source[0])
            self.show_tensor_to_pil(generated_reference[0])

    def load_image_to_tensor(self, file_path: str) -> torch.Tensor:
        image = cv2.imread(file_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        return self.to_tensor(image).unsqueeze(0)

    @staticmethod
    def show_tensor_to_pil(tensor) -> None:
        image = tensor.permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow((0.5 * image + 0.5).clip(0, 1))
        plt.show()

    @staticmethod
    def build_transform() -> transforms:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])


if __name__ == '__main__':
    source_image_path: str = './images/flickr_wild_000004.jpg'
    reference_image_path: str = './images/flickr_cat_000008.jpg'
    pre_trained_model_path: str = './expr/checkpoints/000100_nets_ema.ckpt'

    stargan_v2 = StarGAN_v2(source_image_path=source_image_path,
                            reference_image_path=reference_image_path,
                            pre_trained_model_path=pre_trained_model_path)
    stargan_v2.generate_new_image()







