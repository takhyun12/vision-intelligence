import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class AdaIN_v1:
    def __init__(self, **kwargs: dict) -> None:
        """ Architecture of style transfer network """
        super(AdaIN_v1, self).__init__()
        self.model_name: str = 'AdaIN_v1'
        self.model_version: str = '1.0.0'
        self.content_image_path: str = kwargs['content_image']
        self.style_image_path: str = kwargs['style_image']

        self.vgg_encoder: torch.nn = self.build_encoder_model()
        self.decoder: torch.nn = self.build_decoder_model()

        # Load pre-trained model
        self.vgg_encoder.eval()
        self.decoder.eval()

        vgg_model_path: str = './models/vgg_normalised.pth'
        decoder_model_path: str = './models/decoder.pth'

        self.vgg_encoder.load_state_dict(torch.load(vgg_model_path))
        self.decoder.load_state_dict(torch.load(decoder_model_path))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg_encoder.to(self.device)
        self.decoder.to(self.device)

        self.vgg_encoder = nn.Sequential(*list(self.vgg_encoder.children())[:31])

        self.content_transform = self.build_transform()
        self.style_transform = self.build_transform()

    def style_transfer(self) -> None:
        content = self.content_transform(Image.open(self.content_image_path))
        style = self.style_transform(Image.open(self.style_image_path))

        style = style.to(self.device).unsqueeze(0)
        content = content.to(self.device).unsqueeze(0)

        with torch.no_grad():
            output = self.adain_style_transfer(content, style, alpha=1.0)
        output = output.cpu()

        output_image_path: str = './results/output.png'
        save_image(output, output_image_path)

        output_image = Image.open(output_image_path)
        output_image.show()

    def adain_style_transfer(self, content, style, alpha=1.0):
        assert (0.0 <= alpha <= 1.0)
        content_feature = self.vgg_encoder(content)
        style_feature = self.vgg_encoder(style)
        feature = self.adaptive_instance_normalization(content_feature, style_feature)
        feature = feature * alpha + content_feature * (1 - alpha)
        return self.decoder(feature)

    @staticmethod
    def build_transform(size=512) -> transforms:
        transform_list = list()
        if size != 0:
            transform_list.append(transforms.Resize(size))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
        return transform

    @staticmethod
    def build_encoder_model() -> torch.nn:
        return nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )

    @staticmethod
    def build_decoder_model() -> torch.nn:
        return nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    @staticmethod
    def calc_mean_std(feature: torch.Tensor, eps=1e-5) -> (torch.Tensor, torch.Tensor):
        dimension_size = feature.size()
        assert (len(dimension_size) == 4)
        N, C = dimension_size[:2]
        feat_var: torch.Tensor = feature.view(N, C, -1).var(dim=2) + eps
        feat_std: torch.Tensor = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean: torch.Tensor = feature.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    @staticmethod
    def adaptive_instance_normalization(content_feature: torch.Tensor, style_feature: torch.Tensor) -> torch.Tensor:
        assert (content_feature.size()[:2] == style_feature.size()[:2])
        size = content_feature.size()
        content_mean, content_std = AdaIN_v1.calc_mean_std(content_feature)
        style_mean, style_std = AdaIN_v1.calc_mean_std(style_feature)
        normalized_feature = (content_feature - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feature * style_std.expand(size) + style_mean.expand(size)


if __name__ == '__main__':
    content_image_path: str = './images/game_woman.jpg'
    style_image_path: str = './images/princess.jpg'

    adain = AdaIN_v1(content_image=content_image_path, style_image=style_image_path)
    adain.style_transfer()
