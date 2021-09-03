from PIL import Image
import matplotlib.pyplot as plt
from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from torchvision import transforms as T
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime


class StarGAN_v1:
    def __init__(self, model_config) -> None:
        """ Architecture of style transfer network """
        super(StarGAN_v1, self).__init__()
        self.model_name: str = 'StarGAN_v1'
        self.model_version: str = '1.0.0'

        # Model configurations.
        self.c_dim = model_config.c_dim
        self.c2_dim = model_config.c2_dim
        self.image_size = model_config.image_size
        self.g_conv_dim = model_config.g_conv_dim
        self.d_conv_dim = model_config.d_conv_dim
        self.g_repeat_num = model_config.g_repeat_num
        self.d_repeat_num = model_config.d_repeat_num

        # Training configurations.
        self.dataset = model_config.dataset
        self.g_lr = model_config.g_lr
        self.d_lr = model_config.d_lr
        self.beta1 = model_config.beta1
        self.beta2 = model_config.beta2
        self.selected_attrs = model_config.selected_attrs

        # Test configurations.
        self.test_iters = model_config.test_iters

        # Miscellaneous.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.model_save_dir = model_config.model_save_dir

        # Build the model and tensorboard.
        self.build_model()


    @staticmethod
    def print_network(model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim + self.c2_dim + 2, self.g_repeat_num)  # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim + self.c2_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    @staticmethod
    def denormalized(x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

            c_trg_list = []
            for i in range(c_dim):
                if dataset == 'CelebA':
                    c_trg = c_org.clone()
                    if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                        c_trg[:, i] = 1
                        for j in hair_color_indices:
                            if j != i:
                                c_trg[:, j] = 0
                    else:
                        c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
                elif dataset == 'RaFD':
                    c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

                c_trg_list.append(c_trg.to(self.device))

            return c_trg_list


if __name__ == '__main__':
    args = SimpleNamespace()

    # Training configuration.
    args.dataset = "CelebA"
    args.selected_attrs = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
    args.g_lr = 0.0001
    args.d_lr = 0.0001
    args.beta1 = 0.5
    args.beta2 = 0.999

    # Test configuration.
    args.test_iters = 200000

    # Model configurations.
    args.c_dim = 5
    args.c2_dim = 8
    args.image_size = 256
    args.g_conv_dim = 64
    args.d_conv_dim = 64
    args.g_repeat_num = 6
    args.d_repeat_num = 6

    # Directories.
    args.model_save_dir = "models/stargan_celeba_256"

    # Miscellaneous.
    args.num_workers = 1
    args.mode = "test"

    stargan = StarGAN_v1(args)
    stargan.restore_model(stargan.test_iters)  # 200000 iterations

    image_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = [T.Resize(image_size), T.ToTensor(), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    transform = T.Compose(transform)

    img_path = "./assets/jennie.jpg"
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0).to(device)

    # 원본 이미지 출력
    plt.imshow(stargan.denormalized(image.data.cpu()).squeeze(0).permute(1, 2, 0))

    # [Black_Hair, Blond_Hair, Brown_Hair, Male, Young]
    c_trg = [[0, 1, 0, 0, 1]]
    c_trg = torch.FloatTensor(c_trg).to(device)
    output = stargan.G(image, c_trg)
    plt.imshow(stargan.denormalized(output.data.cpu()).squeeze(0).permute(1, 2, 0))
    plt.show()

