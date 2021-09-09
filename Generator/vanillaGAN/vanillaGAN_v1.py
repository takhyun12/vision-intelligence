import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import warnings
warnings.filterwarnings('ignore')
import time


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]
            if normalize:
                layers.append(nn.BatchNorm1d(output_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 1 * 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1 * 28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        flattened = img.view(img.size(0), -1)
        output = self.model(flattened)
        return output


if __name__ == '__main__':
    latent_dim = 100

    transforms_train = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = datasets.MNIST(root="./dataset", train=True, download=True, transform=transforms_train)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

    # 생성자(generator)와 판별자(discriminator) 초기화
    generator = Generator()
    discriminator = Discriminator()

    generator.cpu()
    discriminator.cpu()

    # 손실 함수(loss function)
    adversarial_loss = nn.BCELoss()
    adversarial_loss.cpu()

    # 학습률(learning rate) 설정
    lr = 0.0002

    # 생성자와 판별자를 위한 최적화 함수
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    n_epochs = 200  # 학습의 횟수(epoch) 설정
    sample_interval = 2000  # 몇 번의 배치(batch)마다 결과를 출력할 것인지 설정
    start_time = time.time()

    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # 진짜(real) 이미지와 가짜(fake) 이미지에 대한 정답 레이블 생성
            real = torch.FloatTensor(imgs.size(0), 1).fill_(1.0)  # 진짜(real): 1
            fake = torch.FloatTensor(imgs.size(0), 1).fill_(0.0)  # 가짜(fake): 0

            real_imgs = imgs.cpu()

            """ 생성자(generator)를 학습합니다. """
            optimizer_G.zero_grad()

            # 랜덤 노이즈(noise) 샘플링
            z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim)).cpu()

            # 이미지 생성
            generated_imgs = generator(z)

            # 생성자(generator)의 손실(loss) 값 계산
            g_loss = adversarial_loss(discriminator(generated_imgs), real)

            # 생성자(generator) 업데이트
            g_loss.backward()
            optimizer_G.step()

            """ 판별자(discriminator)를 학습합니다. """
            optimizer_D.zero_grad()

            # 판별자(discriminator)의 손실(loss) 값 계산
            real_loss = adversarial_loss(discriminator(real_imgs), real)
            fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            # 판별자(discriminator) 업데이트
            d_loss.backward()
            optimizer_D.step()

            done = epoch * len(dataloader) + i
            if done % sample_interval == 0:
                # 생성된 이미지 중에서 25개만 선택하여 5 X 5 격자 이미지에 출력
                save_image(generated_imgs.data[:25], f"result/{done}.png", nrow=5, normalize=True)

        print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s]")
