from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime


class StarGAN_v1:
    def __init__(self, **kwargs: dict) -> None:
        """ Architecture of style transfer network """
        super(StarGAN_v1, self).__init__()
        self.model_name: str = 'StarGAN_v1'
        self.model_version: str = '1.0.0'


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

    print(args)
