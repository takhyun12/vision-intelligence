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

        # configuration
        self.image_size: int = 512
        style_dimensions: int = 768
        domain_number: int = 3

        # model build
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.style_encoder = StyleEncoder(self.image_size, style_dimensions, domain_number).to(self.device)
        self.generator = Generator(self.image_size, style_dimensions, w_hpf=0).to(self.device)

        pre_trained_model = torch.load("./expr/checkpoints/000100_nets_ema.ckpt", map_location=self.device)
        self.generator.load_state_dict(pre_trained_model["generator"])
        self.style_encoder.load_state_dict(pre_trained_model["style_encoder"])
        self.generator.eval()
        self.style_encoder.eval()
        del pre_trained_model

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

        cat = self.load_image("./images/flickr_cat_000008.jpg")
        self.show_tensor(cat[0])
        dog = self.load_image("./images/jindo.jpg")
        self.show_tensor(dog[0])

        with torch.no_grad():
            cat_style = self.style_encoder(cat.to(self.device), torch.LongTensor([1]).to(self.device))
            dog_style = self.style_encoder(dog.to(self.device), torch.LongTensor([0]).to(self.device))
            new_cat = self.generator(cat.to(self.device), dog_style)
            new_dog = self.generator(dog.to(self.device), cat_style)
            self.show_tensor(new_cat[0])
            self.show_tensor(new_dog[0])

    def load_image(self, f_name):
        img = cv2.imread(f_name, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        img = self.to_tensor(img).unsqueeze(0)
        return img

    @staticmethod
    def show_tensor(tensor):
        img = tensor.permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow((0.5 * img + 0.5).clip(0, 1))
        plt.show()


plt.rcParams["figure.figsize"] = (10, 10)
stargan_v2 = StarGAN_v2()





