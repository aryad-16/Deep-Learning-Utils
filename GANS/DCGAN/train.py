from sympy import true
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from model import Discriminator, Generator, initialize_weights

#HYPERPARAMETERS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
NOISE_DIM = 100
IMG_SIZE = 64
CHANNELS_IMG = 3 
BATCH_SIZE = 128
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)],
        [0.5 for _ in range(CHANNELS_IMG)],
    )
])

# dataset = datasets.MNIST(root='/dataset', transform=transforms, download=true, train=true)
dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
loader = DataLoader(dataset, BATCH_SIZE, true)

disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
initialize_weights(disc)
initialize_weights(gen)

opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn((32, NOISE_DIM, 1, 1)).to(device)
writer_fake = SummaryWriter(log_dir='DCGAN/tensorboard_logger/fake')
writer_real = SummaryWriter(log_dir='DCGAN/tensorboard_logger/real')
step=0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, NOISE_DIM, 1, 1)).to(device)

        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        fake = gen(noise)
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake)/2
        disc.zero_grad()
        loss_disc.backward(retain_graph=true)
        opt_disc.step()

        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        ##Tensorboard Logging
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                #Taking 32 examples
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image(
                    "Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Real Images", img_grid_real, global_step=step
                )
            step += 1




